use inference_backends::stablediffusioncpp::{
    StableDiffusionCppConfig, StableDiffusionJob,
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

    let job = StableDiffusionJob::mage_flow_turbo_job()
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

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for (n, outfile) in (0..max_iter)
        .map(|n| format!("mage_flow_1_{:02}", n))
        .enumerate()
    {
        println!("running job {}/{max_iter}", n + 1);
        simple_generation(
            &mut sdcfg,
            &job.clone().with_seed(random()),
            outfile,
            LogSetting::Err(LevelFilter::ERROR),
        )
        .await
        .unwrap()
    }
}
