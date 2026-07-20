use inference_backends::stablediffusioncpp::{
    FlashAttentionMode, Flux2Klein9b, SamplingMethod, Scheduler, StableDiffusionCppConfig,
    StableDiffusionEvent, StableDiffusionJob, ZImageTurboJob,
};

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

const GENERATED_IMAGE_FILE: &str = "flux2klein9b_edit_01_generated.png";
const EDITED_IMAGE_FILE: &str = "flux2klein9b_edit_01_edited.png";

#[tokio::main]
async fn main() {
    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();

    // 1) generate image of an owl
    let generation_job = ZImageTurboJob::default()
        .with_steps(8)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_offload_to_cpu(true)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1024)
        .with_height(1024)
        .with_prompt(
            r#"
high quality photo of a boreal owl sitting on a table inside a stylish coffee shop and drinking a cappuchino.
"#,
        );

    let mut generated_image_data = Vec::new();
    let mut event_receiver = sdcfg
        .run(&generation_job)
        .expect("failed to run generation job");

    while let Some(event) = event_receiver.recv().await {
        match event {
            StableDiffusionEvent::GenerationFinished {
                boxed_data: image_data,
                duration: _,
            } => {
                generated_image_data = *image_data;
                std::fs::write(GENERATED_IMAGE_FILE, &generated_image_data)
                    .expect("failed to write generated image to file");
                println!("image generated and saved as {GENERATED_IMAGE_FILE}");
            }
            StableDiffusionEvent::GenerationStarted {
                seed: _,
                started_at: _,
            } => println!("generating image to edit..."),
            StableDiffusionEvent::Error(e) => panic!("aborting due to error: {e}"),
            StableDiffusionEvent::Killed => panic!("aborting, since sd-cli was stopped"),
            StableDiffusionEvent::Progress {
                step,
                nsteps,
                duration: _,
            } => println!("generating (step {step}/{nsteps})"),
            _ => {}
        }
    }

    // 2) edit generated image

    let uli_reference_image =
        std::fs::read("pose_04_784_816.png").expect("failed to read uli-reference-image");
    let edit_job = Flux2Klein9b::default()
        .with_steps(4)
        .with_cfg_scale(1.0)
        .with_guidance(3.5)
        .with_offload_to_cpu(true)
        .with_flash_attention_mode(FlashAttentionMode::Full)
        .with_scheduler(Scheduler::Simple)
        .with_sampling_method(SamplingMethod::Euler)
        .with_width(1024)
        .with_height(1024)
        .with_ref_image_1(generated_image_data)
        .with_ref_image_2(uli_reference_image)
        .with_prompt(
            r#"
replace the owl with the owl character from reference image 2 preserving everything else.
the owl is holding the mug with its flush wings and stands on its dark brown felted feet.
there are no talons, claws, fingers, hands or paws are shown in the image.
"#,
        );

    sdcfg.stop().await;

    event_receiver = sdcfg.run(&edit_job).expect("failed to run edit job");
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
            _ => {}
        }
    }
}
