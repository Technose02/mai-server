use crate::stablediffusioncpp::{
    StableDiffusionCppConfig, StableDiffusionError, StableDiffusionEvent, StableDiffusionJob,
    StableDiffusionResult,
};
use chrono::NaiveTime;
use std::fmt::Display;

pub async fn simple_generation<J: StableDiffusionJob>(
    config: &StableDiffusionCppConfig,
    job: &J,
    filename: impl Display,
) -> StableDiffusionResult<()> {
    let mut event_receiver = config
        .run(job)
        .map_err(|e| StableDiffusionError::Custom(format!("error running job: {e}")))?;

    while let Some(event) = event_receiver.recv().await {
        match event {
            StableDiffusionEvent::GenerationStarted {
                seed,
                started_at: _,
            } => {
                println!("generation job started with seed {seed}");
            }
            StableDiffusionEvent::Progress {
                step,
                nsteps,
                duration,
            } => {
                println!(
                    "still generating (step {step} of {nsteps} completed, {}ms elapsed)",
                    duration.as_millis()
                )
            }
            StableDiffusionEvent::Error(e) => {
                return Err(StableDiffusionError::Custom(format!(
                    "generation failed with an error: {e}"
                )));
            }
            StableDiffusionEvent::GenerationFinished {
                boxed_data,
                duration,
            } => {
                tokio::fs::write(format!("{filename}.png"), *boxed_data)
                    .await
                    .map_err(|e| {
                        StableDiffusionError::Custom(format!(
                            "error writing generated image to file: {e}"
                        ))
                    })?;
                println!(
                    "generation job finished successfully after {} (see '{filename}.png')",
                    (NaiveTime::from_hms_opt(0, 0, 0)
                        .expect("this static naive-time-definition is valid")
                        + duration)
                        .format("%H:%M:%S")
                );
                break;
            }

            // ignore other output for now
            StableDiffusionEvent::StdErrLine(_) | StableDiffusionEvent::StdOutLine(_) => {}
        }
    }
    Ok(())
}
