//! Transition cost bench: preset load (parse + compile + pipelines), steady-state frame cost,
//! and the per-frame cost while a soft blend runs — same-mode (the MD2 in-pipeline blend) and
//! cross-mode (the image-space transition of two renderers). Native wgpu -> Metal mirrors the
//! browser's compile path closely enough to size the jank sources; run against the app's own
//! pack: `cargo run -p lactea --release --example transition_bench -- <preset-dir> [N]`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use lactea::{Compat, Engine, EngineConfig, FrameClock, Transition};

const W: u32 = 1280;
const H: u32 = 720;
const WARMUP_FRAMES: u32 = 30;
const MEASURE_FRAMES: u32 = 60;

struct FrameMs {
    samples: Vec<f32>,
}

impl FrameMs {
    fn new() -> FrameMs {
        FrameMs { samples: Vec::new() }
    }
    fn add(&mut self, ms: f32) {
        if self.samples.len() < MEASURE_FRAMES as usize {
            self.samples.push(ms);
        }
    }
    fn median(&self) -> f32 {
        let mut s = self.samples.clone();
        if s.is_empty() {
            return 0.0;
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }
    fn max(&self) -> f32 {
        self.samples.iter().cloned().fold(0.0, f32::max)
    }
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    view: wgpu::TextureView,
    frame_clock: u32,
    pcm: Vec<f32>,
}

impl Gpu {
    fn render(&mut self, engine: &mut Engine, fps: &mut FrameMs) {
        // bass-heavy fake audio, one frame's worth at 48 kHz
        self.pcm.clear();
        for i in 0..576 {
            let t = (self.frame_clock as f32 * 576.0 + i as f32) / 48_000.0;
            self.pcm.push(
                0.6 * (t * 2.0 * std::f32::consts::PI * 55.0).sin() * (t * 2.0).sin()
                    + 0.2 * (t * 2.0 * std::f32::consts::PI * 880.0).sin(),
            );
        }
        engine.push_pcm_f32(&self.pcm, 1);
        let t0 = Instant::now();
        let mut enc = self.device.create_command_encoder(&Default::default());
        engine.render(
            &mut enc,
            &self.view,
            FrameClock {
                time: f64::from(self.frame_clock) / 60.0,
                frame: self.frame_clock,
            },
        );
        self.queue.submit(Some(enc.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        fps.add((Instant::now() - t0).as_secs_f32() * 1000.0);
        self.frame_clock += 1;
    }

    fn warm(&mut self, engine: &mut Engine, frames: u32) {
        let mut sink = FrameMs::new();
        for _ in 0..frames {
            self.render(engine, &mut sink);
        }
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: transition_bench <preset-dir> [sample-count]");
    let sample_count: usize = std::env::args()
        .nth(2)
        .and_then(|n| n.parse().ok())
        .unwrap_or(10);

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no wgpu adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bench"),
        ..Default::default()
    }))
    .expect("no device");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = target.create_view(&Default::default());

    let mut engine = Engine::new(
        &device,
        &queue,
        EngineConfig {
            width: W,
            height: H,
            seed: 7,
            compat: Compat::MilkDrop2,
            sample_rate: 48_000,
            fps: 60,
            ..Default::default()
        },
    )
    .unwrap();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("preset dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("milk")))
        .collect();
    files.sort();
    let step = (files.len() / sample_count.max(1)).max(1);
    let picked: Vec<PathBuf> = files.into_iter().step_by(step).take(sample_count).collect();
    println!(
        "{} presets sampled from {dir} ({}x{}, {} frames measured per phase)\n",
        picked.len(),
        W,
        H,
        MEASURE_FRAMES
    );

    let mut gpu = Gpu {
        device,
        queue,
        view,
        frame_clock: 0,
        pcm: Vec::new(),
    };

    // ---- per-preset load cost ---------------------------------------------------------------
    println!("{:<56} {:>9} {:>10}", "preset", "load ms", "steady ms");
    let mut loads = Vec::new();
    let mut steadies = Vec::new();
    for path in &picked {
        let bytes = std::fs::read(path).expect("preset bytes");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let t0 = Instant::now();
        engine
            .load_preset(&bytes, &name, Transition::Cut)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let load_ms = (Instant::now() - t0).as_secs_f32() * 1000.0;
        loads.push(load_ms);

        let mut fps = FrameMs::new();
        gpu.warm(&mut engine, WARMUP_FRAMES);
        for _ in 0..MEASURE_FRAMES {
            gpu.render(&mut engine, &mut fps);
        }
        steadies.push(fps.median());
        println!("{:<56} {:>9.1} {:>10.2}", truncate(&name, 56), load_ms, fps.median());
    }
    loads.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nload : median {:.1} ms, p90 {:.1} ms, max {:.1} ms\n",
        loads[loads.len() / 2],
        loads[(loads.len() as f32 * 0.9) as usize],
        *loads.last().unwrap()
    );

    // ---- reload probe: driver / wgpu cache warmth on an identical second load ---------------
    {
        let bytes = std::fs::read(&picked[0]).unwrap();
        let t0 = Instant::now();
        engine.load_preset(&bytes, "reload-a.milk", Transition::Cut).unwrap();
        let again = (Instant::now() - t0).as_secs_f32() * 1000.0;
        println!("reload of preset[0]: {:.1} ms\n", again);
    }

    // ---- blend cost: same-mode (MD2 in-pipeline) --------------------------------------------
    let a = &picked[0];
    let b = &picked[picked.len() / 2];
    load_cut(&mut gpu, &mut engine, a);
    let mut fps = FrameMs::new();
    gpu.warm(&mut engine, WARMUP_FRAMES);
    let bytes = std::fs::read(b).unwrap();
    engine
        .load_preset(&bytes, "blend-b.milk", Transition::Soft { seconds: 5.0 })
        .unwrap();
    let first_blend = gpu.first_frame_ms(&mut engine, &mut fps);
    for _ in 0..(MEASURE_FRAMES * 4) {
        gpu.render(&mut engine, &mut fps);
    }
    println!(
        "same-mode blend  : first frame {:>7.2} ms, median {:>6.2} ms, max {:>7.2} ms",
        first_blend,
        fps.median(),
        fps.max()
    );

    // ---- blend cost: cross-mode (image-space transition, two renderers) ---------------------
    // a plain preset + `q40` evidence routes to MilkDrop 3, like the app's MD3 pack presets
    let md3 = "MILKDROP_PRESET_VERSION=201\nPSVERSION=2\nPSVERSION_WARP=2\nPSVERSION_COMP=0\nfDecay=1\n\
               fWaveAlpha=0\nper_frame_1=q40 = 1;\n";
    load_cut(&mut gpu, &mut engine, a);
    let mut fps = FrameMs::new();
    gpu.warm(&mut engine, WARMUP_FRAMES);
    engine
        .load_preset_as_compat(
            md3.as_bytes(),
            "md3-bench.milk",
            Transition::Soft { seconds: 5.0 },
            Compat::MilkDrop3,
        )
        .unwrap();
    let first_cross = gpu.first_frame_ms(&mut engine, &mut fps);
    for _ in 0..(MEASURE_FRAMES * 4) {
        gpu.render(&mut engine, &mut fps);
    }
    println!(
        "cross-mode blend : first frame {:>7.2} ms, median {:>6.2} ms, max {:>7.2} ms",
        first_cross,
        fps.median(),
        fps.max()
    );
    // a second cross-mode transition: the transition blender now exists
    load_cut(&mut gpu, &mut engine, a);
    engine
        .load_preset_as_compat(
            md3.as_bytes(),
            "md3-bench-2.milk",
            Transition::Soft { seconds: 5.0 },
            Compat::MilkDrop3,
        )
        .unwrap();
    let mut fps2 = FrameMs::new();
    let second_cross = gpu.first_frame_ms(&mut engine, &mut fps2);
    println!("cross-mode again : first frame {:>7.2} ms", second_cross);
    let _ = steadies;
}

impl Gpu {
    fn first_frame_ms(&mut self, engine: &mut Engine, fps: &mut FrameMs) -> f32 {
        let t0 = Instant::now();
        self.render(engine, fps);
        (Instant::now() - t0).as_secs_f32() * 1000.0
    }
}

fn load_cut(gpu: &mut Gpu, engine: &mut Engine, path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    engine
        .load_preset(&bytes, &name, Transition::Cut)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    gpu.warm(engine, WARMUP_FRAMES);
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n - 1).collect();
        out.push('…');
        out
    }
}
