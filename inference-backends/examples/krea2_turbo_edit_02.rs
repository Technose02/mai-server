use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, SamplingMethod, Scheduler, StableDiffusionCppConfig, StableDiffusionJob,
    helpers::{LogSetting, simple_generation},
};
use rand::random;
use tracing::level_filters::LevelFilter;

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//const VALID_PATH_TO_EXECUTABLE: &str = "/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

#[tokio::main]
async fn main() {
    let mut max_iter = 100;
    if let Some(max) = std::env::args().nth(1).and_then(|s| str::parse(&s).ok()) {
        max_iter = usize::min(max_iter, max);
    }

    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .init();

    let base_img = std::fs::read("/data0/dev/rust/mai-server/krea2_1_03.png")
        .expect("failed to read ref_image");
    let ref_img = std::fs::read("/home/technose02/Pictures/uli_poster/reference_images_for_description/IMG_20260725_053533531_HDR.jpg")
        .expect("failed to read ref_image");

    let job = StableDiffusionJob::krea2_turbo_edit_job()
        .with_steps(8)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1232)
        .with_height(1600)
        .with_init_png(base_img)
        .with_ref_png_1(ref_img)
        .with_prompt(
            r#"
replace the owl with the cappuchino with the owl held in the hand
"#,
        );

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for (n, outfile) in (0..max_iter)
        .map(|n| format!("krea2_turbo_edit_2_{:02}", n))
        .enumerate()
    {
        println!("running job {}/{max_iter}", n + 1);
        simple_generation(
            &mut sdcfg,
            &job.clone().with_seed(random()),
            outfile,
            LogSetting::Err(LevelFilter::INFO),
        )
        .await
        .unwrap()
    }
}
