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

    let job = StableDiffusionJob::krea2_turbo_job()
                .with_width(1232)
                .with_height(1600)
                .with_prompt(r#"
Vertical 1232x1600 portrait composition, eye-level medium close-up, a small seamless bottom-heavy ovoid plush toy owl Uli seated dead-center on a round weathered oak café table, Uli as the dominant central figure occupying at least a third of the total frame area. Uli is one continuous squishable mass with zero separation between head, neck and torso, 15-20cm in scale with a soft beanbag-like weighted mass settling into the tabletop; shaggy long-pile muted taupe and brownish-grey faux fur across the whole body, mottled cream and beige streaks on the belly, short dense fur rings around the eyes, large high-gloss hard plastic black sphere eyes with distinct white specular highlights, a small matte chocolate-brown felt beak, two small rounded chocolate-brown felt pads as feet attached directly to the bottom of the body, and a dark muted forest-green knit scarf constricting the fur so the plush body poofs out around the edges of the fabric; the object is entirely made of fabric and stuffing with NO biological feathers, NO hands, NO fingers, NO claws, NO talons, NO legs, and NO joints. Uli is nestled and pressed into the table surface, his soft unarticulated shaggy wing-flap fold constricting against the porcelain saucer beside him, the plush torso compressed and sinking into the thick faux fur with a deep indentation in the soft stuffing. Beside Uli stands a creamy cappuccino in a thin white ceramic cup on a speckled saucer, micro-textured milk foam, thin wisps of steam rising with soft volumetric diffusion, subsurface scattering glowing through the thin ceramic, one crisp specular highlight on the glossy foam surface.
Behind Uli, a tall window with dark wrought-iron mullions fills the upper half of the frame, rain streaking down the exterior glass, droplets catching warm interior glow, faint soft reflections of the warm room mirrored on the pane. Through the glass, a rainy British autumn street: a narrow brick lane with wet glistening cobblestones, dense canopies of orange amber and rust maple leaves, a small café with a striped awning and round bistro tables, glowing gas street lamps and brass lanterns, warm golden lamplight pooling on the wet stones, distant façades dissolving into depth of field with soft creamy bokeh; the street is completely empty, no pedestrians, no patrons, no human figures anywhere in frame.
The interior stays bright, airy and high-key, never dark: warm tungsten glow from a hanging brass lamp, a small candle flame at the table edge, honey-toned wood, a folded linen napkin and a small glass vase with one dried branch; the indoor surfaces remain dry and clean with no water, no wet patches, no fallen leaves indoors. Warm low-angle rim light traces Uli's shaggy fur silhouette, specular glints in his plastic eyes, soft bounce light lifting the steam above the cappuccino. Muted Portra 400-like color palette, gentle mid-tone contrast, fine subtle film grain, matte-print-friendly tonal range with unclipped highlights, balanced warm tungsten interior against cool rain-washed window light.
"#);

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for (n, outfile) in (0..max_iter)
        .map(|n| format!("krea2_turbo_3_{:02}", n))
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
