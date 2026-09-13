use inference_backends::stablediffusioncpp::{
    StableDiffusionCppConfig, StableDiffusionJob,
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

    let base_img = std::fs::read("/home/technose02/Pictures/uli_poster/20260913/base_672_864.png")
        .expect("failed to read ref_image");
    let ref_img = std::fs::read("/home/technose02/Pictures/uli_poster/20260913/uli_ref_a.png")
        .expect("failed to read ref_image");

    let job = StableDiffusionJob::fire_red_image_edit_8steps_job()
        .with_width(672)
        .with_height(864)
        .with_ref_png_1(base_img)
        .with_ref_png_2(ref_img)
        .with_prompt("Replace the owl in image1 with the owl in image2.");

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for outfile in (0..=100).map(|n| format!("fire_red_image_edit_1_{:02}", n)) {
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
