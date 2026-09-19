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

    let job = StableDiffusionJob::krea2_turbo_job()
                .with_width(1232)
                .with_height(1600)
                .with_prompt(r#"
A cinematic, high-detail lifestyle photograph of a small, artisanal stuffed owl resting on a polished dark walnut coffee shop table. The owl is a seamless, bottom-heavy ovoid shape with no neck, covered in luxurious, long-pile shaggy taupe faux fur. Its belly features mottled cream and light brown streaks. A muted forest green knit scarf is wrapped around its middle, causing the soft fur to poof out charmingly at the edges. The owl has large, high-gloss black plastic eyes that catch the warm, amber ambient light with sharp specular highlights. A small, elegant ceramic cup filled with a cappuccino, featuring delicate latte art, is nestled deeply into the owl's soft, shaggy side; the plush body compresses around the cup, making it look as though the owl is hugging the warmth of the drink. The background is a beautifully blurred, sophisticated cafe interior with soft bokeh, a hint of a wooden bookshelf, and a window showing golden autumn leaves outside. The lighting is warm, cozy, and atmospheric, evoking a sense of calm, adult relaxation. Macro photography style, extreme detail on the fabric textures, ceramic glaze, and wood grain.
"#);

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for (n, outfile) in (0..max_iter)
        .map(|n| format!("krea2_turbo_1_{:02}", n))
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
