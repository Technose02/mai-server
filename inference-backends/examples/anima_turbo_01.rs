use inference_backends::stablediffusioncpp::{
    AnimaTurboJob, SamplingMethod, Scheduler, StableDiffusionCppConfig, StableDiffusionEvent,
    StableDiffusionJob,
};
use std::path::PathBuf;

// ROCM
const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";

// VULKAN
//const VALID_PATH_TO_EXECUTABLE: &str =
//    "/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

async fn generate(outfile: &str, width: usize, height: usize, prompt: &str) {
    let path_to_executable: PathBuf = VALID_PATH_TO_EXECUTABLE.into();
    let sdcfg = StableDiffusionCppConfig::init(path_to_executable)
        .unwrap()
        .with_temporary_output_dir("/tmp")
        .with_flash_attention();

    let mut event_receiver = sdcfg
        .run(
            AnimaTurboJob::default()
                .with_steps(8)
                .with_cfg_scale(1.0)
                .with_guidance(0.0)
                .with_offload_to_cpu(false)
                .with_scheduler(Scheduler::Simple)
                .with_sampling_method(SamplingMethod::Euler)
                .with_width(width)
                .with_height(height)
                .with_prompt(prompt),
        )
        .await
        .unwrap();

    while let Some(event) = event_receiver.recv().await {
        match event {
            StableDiffusionEvent::GenerationStarted {
                seed,
                started_at: _,
            } => {
                println!("generation job started with seed {seed}");
            }
            StableDiffusionEvent::Progress {
                step,
                nsteps,
                duration,
            } => {
                println!(
                    "still generating (step {step} of {nsteps} completed, {}ms elapsed)",
                    duration.as_millis()
                )
            }
            StableDiffusionEvent::Error(e) => {
                panic!("generation failed with an error: {e}");
            }
            StableDiffusionEvent::GenerationFinished {
                boxed_data,
                duration,
            } => {
                tokio::fs::write(format!("{outfile}.png"), *boxed_data)
                    .await
                    .unwrap();
                println!(
                    "generation job finished successfully after {} (see '{outfile}.png')",
                    (chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap() + duration)
                        .format("%H:%M:%S")
                );
                break;
            }

            // ignore other output for now
            StableDiffusionEvent::StdErrLine(_) | StableDiffusionEvent::StdOutLine(_) => {}
        }
    }
}

#[tokio::main]
async fn main() {
    for outfile in (0..=10).map(|n| format!("anima_turbo_1_{:02}", n)) {
        generate(
            &outfile,
            1024,
            1024,
            r#"
Professional 3D character design sheet of an adorable, fluffy baby owl in Disney Pixar art style.
The character, Uli, features extremely soft, voluminous light-brown taupe fur with messy, cute tufts
on top of his head, large expressive glistening dark eyes, and small dark brown rounded feet.
He is wearing a detailed, chunky, hand-crocheted dark green ribbed wool scarf wrapped snugly around his neck.
The image consists of four orthographic views: full front view, profile side view, full back view, and a
charming three-quarter view. High-resolution 8k render, cinematic character design, subsurface scattering
on fur, intricate knit texture on the scarf. Set against a solid, plain white background with no shadows,
no reflections, and no backdrop, completely isolated.
"#
        )
        .await
    }
}
