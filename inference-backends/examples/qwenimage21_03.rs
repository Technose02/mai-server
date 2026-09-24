use image::ImageReader;
use inference_backends::stablediffusioncpp::{
    StableDiffusionCppConfig, StableDiffusionEvent, StableDiffusionJob,
};
use std::{fs::read as read_file, io::Cursor};
use tracing::level_filters::LevelFilter;

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//const VALID_PATH_TO_EXECUTABLE: &str = "/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

const EDITED_IMAGE_FILE: &str = "/home/technose02/Pictures/uli_poster/20260913/uli_ref_a_nobg.png";
const REFERENCE_FILE: &str = "/home/technose02/Pictures/uli_poster/20260913/uli_ref_a.png";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .init();

    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();

    let raw_ref_image = read_file(REFERENCE_FILE).expect("failed to read reference-image");
    let reference_image = ImageReader::new(Cursor::new(&raw_ref_image))
        .with_guessed_format()
        .expect("wrong reference-image format")
        .decode()
        .expect("failed to decode reference-image");

    let edit_job = StableDiffusionJob::qwen_image_21_turbo_job()
        .with_width(reference_image.width() as usize)
        .with_height(reference_image.height() as usize)
        .with_ref_png_1(raw_ref_image)
        .with_prompt("Extract the woman. Output format: RGBA with transparent background.");

    sdcfg.stop().await;

    let mut event_receiver = sdcfg.run(&edit_job).expect("failed to run edit job");
    while let Some(event) = event_receiver.recv().await {
        match event {
            StableDiffusionEvent::GenerationFinished {
                boxed_data: image_data,
                duration: _,
            } => {
                std::fs::write(EDITED_IMAGE_FILE, *image_data)
                    .expect("failed to write edited image to file");
                println!("image generated and saved as {EDITED_IMAGE_FILE}");
            }
            StableDiffusionEvent::GenerationStarted {
                seed: _,
                started_at: _,
            } => println!("editing image..."),
            StableDiffusionEvent::Error(e) => panic!("aborting due to error: {e}"),
            StableDiffusionEvent::Killed => panic!("aborting, since sd-cli was stopped"),
            StableDiffusionEvent::Progress {
                step,
                nsteps,
                duration: _,
            } => println!("editing (step {step}/{nsteps})"),

            StableDiffusionEvent::StdOutLine(out) => println!("[STABLEDIFFUSION::STDOUT] {out}"),
            StableDiffusionEvent::StdErrLine(out) => println!("[STABLEDIFFUSION::STDERR] {out}"),
        }
    }
}
