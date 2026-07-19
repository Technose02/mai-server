mod stablediffusionconfig;
pub use stablediffusionconfig::{StableDiffusionCppConfig, StableDiffusionEvent};

mod stablediffusionjob;
pub use stablediffusionjob::templates::{AnimaTurboJob, Krea2TurboJob, ZImageJob, ZImageTurboJob};
pub use stablediffusionjob::{FlashAttentionMode, SamplingMethod, Scheduler, StableDiffusionJob};

pub mod helpers;

#[derive(Debug)]
pub enum StableDiffusionError {
    Custom(String),
}

impl std::fmt::Display for StableDiffusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StableDiffusionError::Custom(text) => write!(f, "CustomError: {text}"),
        }
    }
}

impl core::error::Error for StableDiffusionError {}

pub type StableDiffusionResult<T> = core::result::Result<T, StableDiffusionError>;
