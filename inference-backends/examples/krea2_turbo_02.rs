use std::time::Duration;

use inference_backends::stablediffusioncpp::{
    Krea2TurboJob, StableDiffusionCppConfig, StableDiffusionEvent,
};

const VALID_PATH_TO_EXECUTABLE: &str =
    "/data0/inference/stable-diffusion.cpp/build-rocm/bin/sd-cli";
//"/data0/inference/stable-diffusion.cpp/build-vulkan/bin/sd-cli";

#[tokio::main]
async fn main() {
    // create sdcfg
    let mut sdcfg =
        StableDiffusionCppConfig::init_with_temp_dir(VALID_PATH_TO_EXECUTABLE, "/tmp").unwrap();

    // run job
    let mut event_receiver = sdcfg.run(&Krea2TurboJob::default()).unwrap();

    // delayed async stop-request to sdcfg
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        sdcfg.stop().await;
    });

    // poll event-receiver
    while let Some(event) = event_receiver.recv().await {
        match event {
            StableDiffusionEvent::Error(e) => panic!("error: {e}"),
            StableDiffusionEvent::GenerationFinished {
                boxed_data: _,
                duration: _,
            } => println!("generation finished"),
            StableDiffusionEvent::GenerationStarted {
                seed: _,
                started_at: _,
            } => println!("generation started"),
            StableDiffusionEvent::Killed => {
                println!("generation aborted")
            }
            StableDiffusionEvent::Progress {
                step: _,
                nsteps: _,
                duration: _,
            }
            | StableDiffusionEvent::StdErrLine(_)
            | StableDiffusionEvent::StdOutLine(_) => {}
        }
    }
}
