//! wasm-bindgen host for the lactea engine as embedded in ApeTapes: a `<canvas>`, a WebGPU
//! surface, PCM and files from JS. Forked from `examples/web`.
//!
//! Deltas from the demo host:
//! - `create` takes the render size explicitly — the app canvas is created detached and the
//!   app owns its `width`/`height` attributes, which are not necessarily set yet at create time.
//! - `load_preset` returns the compat mode the preset actually loaded under, so the app can
//!   badge MilkDrop 3 presets and log triage data.
//!
//! Everything else is the demo host verbatim: the engine crates open no windows, audio devices
//! or files, so the whole host is this file. The app owns `requestAnimationFrame`, audio
//! (an AnalyserNode it reads float samples from), files, and the canvas attributes.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use lactea::{Compat, Engine, EngineConfig, FrameClock, Image, TextureProvider, Transition};
use wasm_bindgen::prelude::*;

/// Textures the page handed over, keyed like [`TextureProvider::load`] names. Shared with the
/// engine's provider so textures added after a preset loaded are found by later loads.
/// `pool` is the subset eligible for the `randNN` sampler — the classic MD2 texture set —
/// while group-folder images (MD3 sprites etc.) are loadable but never enter that pool, so
/// registering them can't shift how MD2-era presets pick random textures.
type TextureMap = Rc<RefCell<TextureStore>>;

struct TextureStore {
    images: HashMap<String, Image>,
    pool: std::collections::BTreeSet<String>,
}

struct MemTextures(TextureMap);

impl TextureProvider for MemTextures {
    fn load(&mut self, name: &str) -> Option<Image> {
        self.0.borrow().images.get(&name.to_ascii_lowercase()).cloned()
    }

    fn random_pool(&self) -> Vec<String> {
        self.0.borrow().pool.iter().cloned().collect()
    }
}

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Lowercase file stem: the name presets use for a texture (`sampler_clouds2` → `clouds2`).
fn texture_key(file_name: &str) -> String {
    let base = file_name.rsplit(['/', '\\']).next().unwrap_or(file_name);
    let stem = base.rsplit_once('.').map_or(base, |(s, _)| s);
    stem.to_ascii_lowercase()
}

/// Short tag for the app: which compat mode a preset actually rendered under.
fn compat_tag(compat: Compat) -> String {
    match compat {
        Compat::ProjectM => "projectm".to_string(),
        Compat::MilkDrop2 => "md2".to_string(),
        Compat::MilkDrop3 => "md3".to_string(),
    }
}

/// The engine plus the WebGPU surface it renders into.
#[wasm_bindgen]
pub struct Lactea {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    engine: Engine,
    textures: TextureMap,
    strong_md3_bands: bool,
}

#[wasm_bindgen]
impl Lactea {
    /// Opens WebGPU on `canvas` and shows the built-in idle preset. `sample_rate` is the rate of
    /// the PCM given to [`Lactea::push_pcm`]; `width`/`height` are the render size (the canvas
    /// element's attributes are the app's business, not ours). `compat` is `"auto"` (per-preset
    /// detection), `"md2"`, `"md3"` or `"projectm"` — the engine's fixed mode for `"md2"` and
    /// friends also selects that mode's audio analysis (MilkDrop 2 → BeatDrop's Hz bands).
    /// `canvas` may be an `HTMLCanvasElement` or an `OffscreenCanvas` (the worker host renders
    /// into its own offscreen).
    pub async fn create(
        canvas: JsValue,
        sample_rate: u32,
        seed: u32,
        width: u32,
        height: u32,
        compat: &str,
    ) -> Result<Lactea, JsError> {
        static HOOK: std::sync::Once = std::sync::Once::new();
        HOOK.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
            }));
        });
        let (width, height) = (width.max(1), height.max(1));
        use wasm_bindgen::JsCast;
        let instance = wgpu::Instance::default();
        let surface_target = if let Ok(c) = canvas.clone().dyn_into::<web_sys::HtmlCanvasElement>() {
            wgpu::SurfaceTarget::Canvas(c)
        } else if let Ok(o) = canvas.dyn_into::<web_sys::OffscreenCanvas>() {
            wgpu::SurfaceTarget::OffscreenCanvas(o)
        } else {
            return Err(js_err("canvas is neither an HTMLCanvasElement nor an OffscreenCanvas"));
        };
        let surface = instance
            .create_surface(surface_target)
            .map_err(js_err)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(js_err)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("lactea web"),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(js_err)?;
        // a validation error from one preset must not take the page down
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            web_sys::console::error_1(&JsValue::from_str(&format!("wgpu error: {e}")));
        }));

        let caps = surface.get_capabilities(&adapter);
        // the engine's output is not sRGB-encoded: prefer a linear surface format
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| js_err("the canvas reports no surface formats"))?;
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| js_err("the canvas is not supported by the adapter"))?;
        config.format = format;
        if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        }
        surface.configure(&device, &config);

        let cfg = EngineConfig {
            width,
            height,
            seed: u64::from(seed),
            compat: match compat {
                "md2" | "auto-md2" => Compat::MilkDrop2,
                "md3" => Compat::MilkDrop3,
                _ => Compat::ProjectM,
            },
            auto_compat: compat == "auto",
            output_format: format,
            sample_rate,
            // browser host: MD3 presets read BeatDrop's kick-aware bands (MD3's authored thirds
            // have half the per-beat contrast and a vocal-dominated "bass") — the capture itself
            // stays reference
            strong_md3_bands: true,
            ..Default::default()
        };
        let mut engine = Engine::new(&device, &queue, cfg).map_err(js_err)?;
        let textures: TextureMap = Rc::new(RefCell::new(TextureStore {
            images: HashMap::new(),
            pool: Default::default(),
        }));
        engine.set_texture_provider(Box::new(MemTextures(textures.clone())));
        engine.load_idle_preset().map_err(js_err)?;
        Ok(Lactea {
            device,
            queue,
            surface,
            config,
            engine,
            textures,
            strong_md3_bands: true,
        })
    }

    /// Loads a `.milk` / `.milk2` preset (the file's bytes). `blend_seconds > 0` blends from the
    /// running preset, otherwise it cuts. `compat` is `"auto"` (per-preset detection), `"md2"`,
    /// `"md3"` or `"projectm"`; a forced mode different from the running one cannot use the mode's
    /// own blend — between MilkDrop 2 and MilkDrop 3 the soft request becomes the image-space
    /// transition of the two renderers, any other mode change is a hard cut. On error the running
    /// preset stays. Returns the compat mode the preset loaded under (`projectm` | `md2` | `md3`).
    pub fn load_preset(
        &mut self,
        bytes: &[u8],
        name: &str,
        blend_seconds: f64,
        compat: &str,
    ) -> Result<String, JsError> {
        let transition = if blend_seconds > 0.0 {
            Transition::Soft {
                seconds: blend_seconds,
            }
        } else {
            Transition::Cut
        };
        let result = match compat {
            // The engine's own detector decides MD3 evidence; everything else runs MilkDrop 2
            // — never projectM, whose fixed audio bands flatten MD2-era presets.
            "auto-md2" => {
                let want = if lactea::compat_for_preset(bytes) == Compat::MilkDrop3 {
                    Compat::MilkDrop3
                } else {
                    Compat::MilkDrop2
                };
                self.engine.load_preset_as_compat(bytes, name, transition, want)
            }
            "md2" => self
                .engine
                .load_preset_as_compat(bytes, name, transition, Compat::MilkDrop2),
            "md3" => self
                .engine
                .load_preset_as_compat(bytes, name, transition, Compat::MilkDrop3),
            "projectm" => self
                .engine
                .load_preset_as_compat(bytes, name, transition, Compat::ProjectM),
            _ => self.engine.load_preset(bytes, name, transition),
        };
        result
            .map(|report| compat_tag(report.compat))
            .map_err(js_err)
    }

    /// Registers an image (PNG / JPEG / TGA / BMP bytes) as the texture named after the file's
    /// stem, for presets loaded afterwards. The argument may be a full path; the stem keys it.
    /// Undecodable formats (e.g. some GIFs) degrade to an unresolved-texture placeholder —
    /// skip at the call site.
    pub fn add_texture(&mut self, file_name: &str, bytes: &[u8]) -> Result<(), JsError> {
        let image = lactea_host::decode_image(bytes).map_err(js_err)?;
        self.textures
            .borrow_mut()
            .images
            .insert(texture_key(file_name), image);
        Ok(())
    }

    /// Like [`Lactea::add_texture`], and the texture also joins the `randNN` random pool —
    /// for the shared MD2 texture set, not per-group sprites.
    pub fn add_pool_texture(&mut self, file_name: &str, bytes: &[u8]) -> Result<(), JsError> {
        let image = lactea_host::decode_image(bytes).map_err(js_err)?;
        let key = texture_key(file_name);
        let mut store = self.textures.borrow_mut();
        store.images.insert(key.clone(), image);
        store.pool.insert(key);
        Ok(())
    }

    /// Feeds interleaved float PCM (`-1..1`) at the rate given to [`Lactea::create`].
    pub fn push_pcm(&mut self, interleaved: &[f32], channels: usize) {
        self.engine.push_pcm_f32(interleaved, channels.max(1));
    }

    /// Debug: the six band values of the last rendered frame, `[bass, mid, treb, bass_att,
    /// mid_att, treb_att]`, from the regime the active preset was authored against.
    pub fn debug_audio(&self) -> Vec<f32> {
        let f = self.engine.audio_frame();
        vec![f.bass, f.mid, f.treb, f.bass_att, f.mid_att, f.treb_att]
    }

    /// Debug: `[strong_md3_bands, active_compat_tag, engine_compat_tag]` — what the feed log
    /// needs to attribute a band reading to a regime. Tags: 0 projectM, 1 md2, 2 md3.
    pub fn debug_flags(&self) -> Vec<f32> {
        let tag = |c: Compat| match c {
            Compat::ProjectM => 0.0,
            Compat::MilkDrop2 => 1.0,
            Compat::MilkDrop3 => 2.0,
        };
        vec![
            if self.strong_md3_bands { 1.0 } else { 0.0 },
            tag(self.engine.renderer().compat()),
            tag(self.engine.current_compat()),
        ]
    }

    /// Mouse position in `0..1` canvas coordinates (y down); `undefined` when outside.
    pub fn set_mouse(&mut self, x: Option<f32>, y: Option<f32>, left_down: bool) {
        self.engine.set_mouse(lactea::MouseState {
            position: x.zip(y).map(|(x, y)| [x, y]),
            left_down,
        });
    }

    /// Resizes the surface and the engine's images (no-op for an unchanged or empty size).
    /// Never touches the canvas element's attributes — the app owns those.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsError> {
        if width == 0 || height == 0 || (width, height) == (self.config.width, self.config.height) {
            return Ok(());
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.engine.resize(width, height).map_err(js_err)
    }

    /// Renders and presents one frame at `time` seconds.
    pub fn render(&mut self, time: f64, frame: u32) -> Result<(), JsError> {
        use wgpu::CurrentSurfaceTexture as C;
        let texture = match self.surface.get_current_texture() {
            C::Success(t) | C::Suboptimal(t) => t,
            C::Outdated | C::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            C::Timeout | C::Occluded => return Ok(()),
            C::Validation => return Err(js_err("surface validation error")),
        };
        let view = texture.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.engine
            .render(&mut encoder, &view, FrameClock { time, frame });
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(texture);
        Ok(())
    }
}
