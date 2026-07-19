use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, SamplingMethod, Scheduler, StableDiffusionCppConfig, StableDiffusionJob,
    ZImageTurboJob, helpers::simple_generation,
};
use std::path::PathBuf;

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

#[tokio::main]
async fn main() {
    let path_to_executable: PathBuf = VALID_PATH_TO_EXECUTABLE.into();
    let sdcfg = StableDiffusionCppConfig::init(path_to_executable)
        .unwrap()
        .with_temporary_output_dir("/tmp")
        .with_flash_attention_mode(FlashAttentionMode::All);

    let job = ZImageTurboJob::default()
                .with_steps(8)
                .with_cfg_scale(1.0)
                .with_guidance(3.5)
                .with_offload_to_cpu(false)
                .with_scheduler(Scheduler::Simple)
                .with_sampling_method(SamplingMethod::Euler)
                .with_width(1024)
                .with_height(1024)
                .with_prompt(r#"
Professional 3D character design sheet of an adorable, fluffy baby owl in Disney Pixar art style.
The character, Uli, features extremely soft, voluminous light-brown taupe fur with messy, cute tufts
on top of his head, large expressive glistening dark eyes, and small dark brown rounded feet.
He is wearing a detailed, chunky, hand-crocheted dark green ribbed wool scarf wrapped snugly around his neck.
The image consists of four orthographic views: full front view, profile side view, full back view, and a
charming three-quarter view. High-resolution 8k render, cinematic character design, subsurface scattering
on fur, intricate knit texture on the scarf. Set against a solid, plain white background with no shadows,
no reflections, and no backdrop, completely isolated.
"#);

    for outfile in (0..=100).map(|n| format!("zimage_turbo_1_{:02}", n)) {
        simple_generation(&sdcfg, &job, outfile).await.unwrap()
    }
}
