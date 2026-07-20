use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, Flux2Klein9b, SamplingMethod, Scheduler, StableDiffusionCppConfig,
    StableDiffusionJob, helpers::simple_generation,
};

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

#[tokio::main]
async fn main() {
    let job = Flux2Klein9b::default()
        .with_steps(4)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_offload_to_cpu(true)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1024)
        .with_height(1024)
        .with_prompt(
            r#"
a lovely cat
"#,
        );

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for outfile in (0..=100).map(|n| format!("flux2klein9b_1_{:02}", n)) {
        simple_generation(&mut sdcfg, &job, outfile).await.unwrap()
    }
}
