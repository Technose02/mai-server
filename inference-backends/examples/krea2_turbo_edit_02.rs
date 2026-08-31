use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, Krea2TurboEditJob, SamplingMethod, Scheduler, StableDiffusionCppConfig,
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

    let base_img = std::fs::read("/data0/dev/rust/mai-server/krea2_1_03.png")
        .expect("failed to read ref_image");
    let ref_img = std::fs::read("/home/technose02/Pictures/uli_poster/reference_images_for_description/IMG_20260725_053533531_HDR.jpg")
        .expect("failed to read ref_image");

    let job = Krea2TurboEditJob::default()
        .with_steps(8)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1232)
        .with_height(1600)
        .with_init_image(base_img)
        .with_ref_image_1(ref_img)
        .with_prompt(
            r#"
replace the owl with the cappuchino with the owl held in the hand
"#,
        );

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for outfile in (0..=100).map(|n| format!("krea2_turbo_edit_2_{:02}", n)) {
        simple_generation(
            &mut sdcfg,
            &job,
            outfile,
            LogSetting::Err(LevelFilter::INFO),
        )
        .await
        .unwrap()
    }
}
