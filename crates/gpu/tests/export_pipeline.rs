//! End-to-end GPU export parity: the registered `filmcraft-gpu` frame renderer drives the full
//! export pipeline (`filmcraft_export::export` → PNG sequence) and its frames must match the
//! `GpuRendering::Off` (CPU reference) frames within the GPU-vs-CPU parity tolerance — p99 of the
//! per-pixel max RGB difference over black ≤ 6 and mean < 1.5 (the golden-test criterion).

use filmcraft_export::{ExportSettings, Format, FrameRenderer, GpuRenderStats, GpuRendering, Progress, export, register_frame_renderer};
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{Generator, MediaSource};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, SequenceSettings, TrackKind};
use filmcraft_render::SourceMap;
use filmcraft_time::{TICKS_PER_SECOND, TimeRange};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static RENDERER_CREATIONS: AtomicUsize = AtomicUsize::new(0);
static PREPARED_CALLS: AtomicUsize = AtomicUsize::new(0);
static LEGACY_CALLS: AtomicUsize = AtomicUsize::new(0);
static PREPARED_ENABLED: AtomicBool = AtomicBool::new(true);
// Four adjacent clips, six frames each: exercise planning across clip boundaries.
const CLIPS: i64 = 4;
const FRAMES_PER_CLIP: i64 = 6;
const FRAMES: i64 = CLIPS * FRAMES_PER_CLIP;

/// App/CLI adapter with dispatch counters and a capability-disabled control.
struct GpuFrameRenderer(filmcraft_gpu::ExportRenderer);

impl FrameRenderer for GpuFrameRenderer {
    fn supports_prepared_plan(&self) -> bool {
        PREPARED_ENABLED.load(Ordering::SeqCst)
    }

    fn render_prepared(&mut self, plan: &filmcraft_render::plan::FramePlan) -> Result<Option<filmcraft_render::Image>, String> {
        PREPARED_CALLS.fetch_add(1, Ordering::SeqCst);
        self.0.render_prepared(plan)
    }

    fn render(
        &mut self,
        project: &filmcraft_project::Project,
        seq: filmcraft_project::ItemId,
        t: filmcraft_time::Tick,
        opts: filmcraft_render::RenderOptions,
        sources: &dyn filmcraft_render::SourceProvider,
    ) -> Result<Option<filmcraft_render::Image>, String> {
        LEGACY_CALLS.fetch_add(1, Ordering::SeqCst);
        self.0.render(project, seq, t, opts, sources)
    }
}

fn project(width: u32, height: u32) -> (Arc<filmcraft_project::Project>, filmcraft_project::ItemId, SourceMap) {
    let mut p = filmcraft_project::Project::new("x");
    let g = GeneratorSource::new(
        Generator::ColorMatte { color: [0.2, 0.5, 0.9, 1.0] },
        320,
        180,
        filmcraft_time::FrameRate::FPS_24,
        filmcraft_time::Tick(2 * TICKS_PER_SECOND),
    );
    let info = g.info().clone();
    let red = p.add_item(
        &info.name.clone(),
        Label::Iris,
        ItemKind::Media(Box::new(MediaClip {
            media: MediaRef::Generator(g.generator.clone()),
            info,
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        })),
        None,
    );
    let seq = p.new_sequence("s", SequenceSettings { width, height, frame_rate: filmcraft_time::FrameRate::FPS_24, ..Default::default() }, 1, 1, None);
    let v = p
        .make_track_item(
            red,
            TrackKind::Video,
            filmcraft_time::Tick::ZERO,
            TimeRange::new(filmcraft_time::Tick::ZERO, filmcraft_time::FrameRate::FPS_24.tick_of(FRAMES_PER_CLIP)),
            filmcraft_time::FrameRate::FPS_24,
        )
        .unwrap();
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    for i in 1..CLIPS {
        let clip = p
            .make_track_item(
                red,
                TrackKind::Video,
                filmcraft_time::FrameRate::FPS_24.tick_of(i * FRAMES_PER_CLIP),
                TimeRange::new(filmcraft_time::Tick::ZERO, filmcraft_time::FrameRate::FPS_24.tick_of(FRAMES_PER_CLIP)),
                filmcraft_time::FrameRate::FPS_24,
            )
            .unwrap();
        p.sequence_mut(seq).unwrap().video_tracks[0].items.push(clip);
    }
    let mut m = SourceMap::default();
    m.0.insert(red, Arc::new(g));
    (Arc::new(p), seq, m)
}

fn png(dir: &str, name: &str) -> Vec<u8> {
    // image_sequence_path flattens the path stem: <stem><index>.png in the temp dir itself.
    let index: u64 = name.trim_start_matches("img").trim_end_matches(".png").parse().unwrap_or(0);
    std::fs::read(std::env::temp_dir().join(format!("fc-gpu-export-{dir}{index:03}.png"))).unwrap()
}

fn export_png(dir: &str, gpu: GpuRendering, (w, h): (u32, u32)) {
    let (p, seq, m) = project(w, h);
    let mut s = ExportSettings {
        format: Format::PngSequence,
        path: std::env::temp_dir().join(format!("fc-gpu-export-{dir}")).to_string_lossy().into_owned(),
        ..Default::default()
    };
    s.range = Some(TimeRange::new(filmcraft_time::Tick::ZERO, filmcraft_time::FrameRate::FPS_24.tick_of(FRAMES)));
    s.gpu_rendering = gpu;
    export(&p, seq, &s, &m, &Progress::default()).unwrap();
}

fn assert_parity(cpu_png: &[u8], gpu_png: &[u8], what: &str) {
    let a = image::load_from_memory(cpu_png).unwrap().to_rgba8();
    let b = image::load_from_memory(gpu_png).unwrap().to_rgba8();
    assert_eq!(a.dimensions(), b.dimensions(), "{what}: size mismatch");
    let d: Vec<u32> = a
        .as_raw()
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_raw().as_chunks::<4>().0)
        .map(|(x, y)| (0..3).map(|k| (x[k] as u32).abs_diff(y[k] as u32)).max().unwrap_or(0))
        .collect();
    let mut d = d;
    d.sort_unstable();
    let p99 = d[d.len() * 99 / 100];
    let mean = d.iter().sum::<u32>() as f64 / d.len() as f64;
    eprintln!("{what}: GPU vs CPU p99 {p99}, mean {mean:.3}");
    assert!(p99 <= 6 && mean < 1.5, "{what}: GPU parity p99 {p99}, mean {mean:.3}");
}

#[test]
fn gpu_export_matches_cpu_and_counts() {
    let Some(_) = filmcraft_gpu::ExportRenderer::new() else {
        eprintln!("SKIPPED (gpu export pipeline): no adapter");
        return;
    };
    // Clean registry first: Auto with no factory must be byte-equal to Off.
    export_png("off-cpu", GpuRendering::Off, (320, 180));
    export_png("off-auto", GpuRendering::Auto, (320, 180));
    assert_eq!(png("off-cpu", "img000.png"), png("off-auto", "img000.png"), "Auto with no registered renderer must be byte-equal to Off");

    register_frame_renderer(|| {
        let n = RENDERER_CREATIONS.fetch_add(1, Ordering::SeqCst) + 1;
        eprintln!("GPU renderer created: {n}");
        Some(Box::new(GpuFrameRenderer(filmcraft_gpu::ExportRenderer::new().expect("adapter present"))))
    });
    let sizes = [(320u32, 180u32), (17, 13), (1, 1), (257, 129)];
    for size in sizes {
        export_png("cpu", GpuRendering::Off, size);
        export_png("gpu", GpuRendering::Auto, size);
        for f in (0..FRAMES).map(|n| format!("{n:03}")) {
            assert_parity(&png("cpu", &format!("img{f}.png")), &png("gpu", &format!("img{f}.png")), &format!("{:?} frame {f}", size));
        }
    }
    // Same pool policy as Pipeline::new; no assumption about host worker count.
    let per_export = rayon::current_num_threads().clamp(2, 4);
    assert_eq!(RENDERER_CREATIONS.load(Ordering::SeqCst), sizes.len() * per_export);
    let expected_frames = sizes.len() * usize::try_from(FRAMES).unwrap();
    assert_eq!(PREPARED_CALLS.load(Ordering::SeqCst), expected_frames);
    assert_eq!(LEGACY_CALLS.load(Ordering::SeqCst), 0);
    let GpuRenderStats { frames, .. } = filmcraft_export::gpu_render_stats();
    assert_eq!(frames, expected_frames as u64);

    PREPARED_ENABLED.store(false, Ordering::SeqCst);
    export_png("legacy", GpuRendering::Auto, sizes[0]);
    assert_eq!(PREPARED_CALLS.load(Ordering::SeqCst), expected_frames, "disabled capability must bypass prepared dispatch");
    assert_eq!(LEGACY_CALLS.load(Ordering::SeqCst), usize::try_from(FRAMES).unwrap());
    assert_eq!(RENDERER_CREATIONS.load(Ordering::SeqCst), (sizes.len() + 1) * per_export);
    export_png("legacy-cpu", GpuRendering::Off, sizes[0]);
    for n in 0..FRAMES {
        let name = format!("img{n:03}.png");
        assert_parity(&png("legacy-cpu", &name), &png("legacy", &name), &format!("legacy frame {n}"));
    }
}
