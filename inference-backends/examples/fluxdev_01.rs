use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, FluxDevJob, SamplingMethod, Scheduler, StableDiffusionCppConfig,
    StableDiffusionJob,
    helpers::{LogSetting, simple_generation},
};
use tracing::level_filters::LevelFilter;

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .init();

    let job = FluxDevJob::default()
        .with_steps(20)
        .with_cfg_scale(1.0)
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
    for outfile in (0..=100).map(|n| format!("fluxdev_1_{:02}", n)) {
        simple_generation(
            &mut sdcfg,
            &job,
            outfile,
            LogSetting::Err(LevelFilter::ERROR),
        )
        .await
        .unwrap()
    }
}
