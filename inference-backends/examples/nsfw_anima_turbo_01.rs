use inference_backends::stablediffusioncpp::{
    AnimaTurboJob, FlashAttentionMode, SamplingMethod, Scheduler, StableDiffusionCppConfig,
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

    let job = AnimaTurboJob::default()
        .with_steps(8)
        .with_cfg_scale(1.0)
        .with_guidance(0.0)
        .with_offload_to_cpu(false)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1280)
        .with_height(720)
        .with_prompt(prompt(1));

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();
    for outfile in (0..=100).map(|n| format!("nsfw_anima_1_{:02}", n)) {
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

fn prompt(variant: usize) -> &'static str {
    match variant {
        0 => {
            r#"nsfw;
disney pixar 3d animation art style;
a modern office with plants, a desk, framed family picture on desk showing husband and children, a chair, a switched on monitor, a door, a large window, rain outside, night time;
two people: [W] and [M], western look;
[W] is a hot academic lady aged 40, long light brown hair, office dress, neglace, earrings, black highheels, short checkered skirt push up above hips, glasses, ripped blouse, small tits, nipples, ripped black pantyhose, pubic hair;
[M] is her boss, male, aged 24, athletic, business man;
[M] is fucking [W] from behind on an office desk, insertion, hardcore;
[M] is passionate and horny;
[W] is surprised and shocked as well as enjoying silently;
only [W] and [M] visible;
shot frontal, camera looking straight into face of [W];
anatomically correct legs, arms and hands;"#
        }
        1 => {
            r#"nsfw;
high quality manga art style;
a modern office with plants, a desk, framed family picture on desk showing husband and children, a chair, a switched on monitor, a door, a large window, rain outside, night time;
two people: [W] and [M];
[W] is a beautiful confident academic lady aged 40, long hair, office dress, neglace, earrings, black highheels, short checkered skirt push up above hips, glasses, ripped blouse, small tits, nipples, ripped black pantyhose, pubic hair;
[M] is her boss, male, aged 24, athletic, business man;
[M] is fucking [W] from behind on an office desk, insertion, hardcore;
[M] is passionate and horny;
[W] is surprised and shocked as well as enjoying silently;
only [W] and [M] visible;
shot frontal, camera looking straight into face of [W];
anatomically correct legs, arms and hands;"#
        }
        _ => "",
    }
}
