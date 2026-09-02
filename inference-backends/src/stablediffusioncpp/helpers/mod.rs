use crate::stablediffusioncpp::{
    StableDiffusionCppConfig, StableDiffusionError, StableDiffusionEvent, StableDiffusionJob,
    StableDiffusionResult,
};
use chrono::NaiveTime;
use std::fmt::Display;
use tracing::level_filters::LevelFilter;

mod logsetting;
pub use logsetting::LogSetting;

mod rescalc;
pub use rescalc::Rescalc;

pub async fn simple_generation(
    config: &mut StableDiffusionCppConfig,
    job: &StableDiffusionJob,
    filename: impl Display,
    log_setting: LogSetting,
) -> StableDiffusionResult<()> {
    let mut event_receiver = config
        .run(job)
        .map_err(|e| StableDiffusionError::Custom(format!("error running job: {e}")))?;

    while let Some(event) = event_receiver.recv().await {
        match (event, &log_setting) {
            (
                StableDiffusionEvent::GenerationStarted {
                    seed,
                    started_at: _,
                },
                _,
            ) => {
                println!("generation job started with seed {seed}");
            }
            (
                StableDiffusionEvent::Progress {
                    step,
                    nsteps,
                    duration,
                },
                _,
            ) => {
                println!(
                    "still generating (step {step} of {nsteps} completed, {}ms elapsed)",
                    duration.as_millis()
                )
            }
            (StableDiffusionEvent::Error(e), _) => {
                return Err(StableDiffusionError::Custom(format!(
                    "generation failed with an error: {e}"
                )));
            }
            (
                StableDiffusionEvent::GenerationFinished {
                    boxed_data,
                    duration,
                },
                _,
            ) => {
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
            (StableDiffusionEvent::Killed, _) => {
                println!("sd-cli process killed");
                break;
            }

            (StableDiffusionEvent::StdErrLine(err_line), log_setting) => match log_setting {
                LogSetting::Err(LevelFilter::ERROR)
                | LogSetting::Both {
                    err_level: LevelFilter::ERROR,
                    out_level: _,
                } => tracing::error!("[stable-diffusion.cpp]: {err_line}"),
                LogSetting::Err(LevelFilter::TRACE)
                | LogSetting::Both {
                    err_level: LevelFilter::TRACE,
                    out_level: _,
                } => tracing::trace!("[stable-diffusion.cpp]: {err_line}"),
                LogSetting::Err(LevelFilter::INFO)
                | LogSetting::Both {
                    err_level: LevelFilter::INFO,
                    out_level: _,
                } => tracing::info!("[stable-diffusion.cpp]: {err_line}"),
                LogSetting::Err(LevelFilter::DEBUG)
                | LogSetting::Both {
                    err_level: LevelFilter::DEBUG,
                    out_level: _,
                } => tracing::debug!("[stable-diffusion.cpp]: {err_line}"),
                LogSetting::Err(LevelFilter::WARN)
                | LogSetting::Both {
                    err_level: LevelFilter::WARN,
                    out_level: _,
                } => tracing::warn!("[stable-diffusion.cpp]: {err_line}"),
                _ => {}
            },

            (StableDiffusionEvent::StdOutLine(outline), log_setting) => match log_setting {
                LogSetting::Out(LevelFilter::ERROR)
                | LogSetting::Both {
                    out_level: LevelFilter::ERROR,
                    err_level: _,
                } => tracing::error!("[stable-diffusion.cpp]: {outline}"),
                LogSetting::Out(LevelFilter::TRACE)
                | LogSetting::Both {
                    out_level: LevelFilter::TRACE,
                    err_level: _,
                } => tracing::trace!("[stable-diffusion.cpp]: {outline}"),
                LogSetting::Out(LevelFilter::INFO)
                | LogSetting::Both {
                    out_level: LevelFilter::INFO,
                    err_level: _,
                } => tracing::info!("[stable-diffusion.cpp]: {outline}"),
                LogSetting::Out(LevelFilter::DEBUG)
                | LogSetting::Both {
                    out_level: LevelFilter::DEBUG,
                    err_level: _,
                } => tracing::debug!("[stable-diffusion.cpp]: {outline}"),
                LogSetting::Out(LevelFilter::WARN)
                | LogSetting::Both {
                    out_level: LevelFilter::WARN,
                    err_level: _,
                } => tracing::warn!("[stable-diffusion.cpp]: {outline}"),
                _ => {}
            },
        }
    }
    Ok(())
}
