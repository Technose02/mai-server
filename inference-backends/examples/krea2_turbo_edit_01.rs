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

    let ref_img_1_data = std::fs::read("/home/technose02/Pictures/uli_poster/reference_images_for_description/IMG_20260725_053533531_HDR.jpg")
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
                .with_ref_png_1(ref_img_1_data)
                .with_prompt(r#"
create an image of the stuffed owl from the image sitting on a rustic wooden table in a cozy, warm autumn coffeeshop. The owl is drinking a small ceramic cappuccino with latte art, holding the cup using a "cuddle grip," clamping the vessel securely between the soft fold of its shaggy wing-flap and its plump, mottled chest, with the cup deeply recessed into the fur. It is wearing the dark autumn-green ribbed knit scarf, tied in a soft, chunky knot at the front with the brownish-taupe ends draping asymmetrically over the cream-and-brown chest fur. The scene is captured in a sophisticated, relaxed adult aesthetic with warm golden-hour lighting, soft bokeh of a rain-streaked window in the background showing fallen orange leaves, and a color palette of deep ambers, forest greens, and muted taupes. High-fidelity textures on the glossy plastic eyes and the tactile, ribbed yarn of the scarf.
"#);

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for (n, outfile) in (0..max_iter)
        .map(|n| format!("krea2_turbo_edit_1_{:02}", n))
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
