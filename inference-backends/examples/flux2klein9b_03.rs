use std::io::Cursor;

use image::{
    GenericImageView,
    ImageFormat::{Jpeg, Png},
    imageops::FilterType::Lanczos3,
};
use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, Flux2Klein9bJob, SamplingMethod, Scheduler, StableDiffusionCppConfig,
    StableDiffusionJob,
    helpers::{LogSetting, Rescalc, simple_generation},
};
use tracing::level_filters::LevelFilter;

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

const BASE_IMAGE: &str = "/data0/dev/rust/mai-server/krea2_1_03.png";
const REFERENCE_IMAGE: &str = "/home/technose02/Pictures/uli_poster/reference_images_for_description/IMG_20260725_053533531_HDR.jpg";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .init();

    let base_image = std::fs::read(BASE_IMAGE).expect("failed to read base_image");
    let reference_image = std::fs::read(REFERENCE_IMAGE).expect("failed to read reference_image");

    let mut base_image =
        image::load(std::io::Cursor::new(base_image), Png).expect("failed to load base_image");
    let mut reference_image = image::load(std::io::Cursor::new(reference_image), Jpeg)
        .expect("failed to load base_image");

    let (mut w_base, mut h_base) = base_image.dimensions();
    let (res, _, _) =
        Rescalc::fit_first(w_base as f32 / h_base as f32, 1024.0 * 1024.0, 0.05, 0.05).unwrap();
    base_image = base_image.resize_exact(res.width, res.height, Lanczos3);
    w_base = res.width;
    h_base = res.height;

    let (w_ref, h_ref) = reference_image.dimensions();
    let (res, _, _) =
        Rescalc::fit_first(w_ref as f32 / h_ref as f32, 1024.0 * 1024.0, 0.05, 0.05).unwrap();
    reference_image = reference_image.resize_exact(res.width, res.height, Lanczos3);

    let base_image = {
        let inner = Vec::<u8>::new();
        let mut c = Cursor::new(inner);
        base_image
            .write_to(&mut c, Png)
            .expect("failed to convert base_image");
        c.into_inner()
    };

    let reference_image = {
        let inner = Vec::<u8>::new();
        let mut c = Cursor::new(inner);
        reference_image
            .write_to(&mut c, Png)
            .expect("failed to convert reference_image");
        c.into_inner()
    };

    let job = Flux2Klein9bJob::default()
        .with_steps(4)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(w_base as usize)
        .with_height(h_base as usize)
        .with_ref_image_1(base_image)
        .with_ref_image_2(reference_image)
        .with_prompt(
            r#"
replace the owl in image1 with the owl in image_2
"#,
        );

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();

    simple_generation(
        &mut sdcfg,
        &job,
        "fluxklein9b_3_01",
        LogSetting::Err(LevelFilter::INFO),
    )
    .await
    .unwrap()
}
