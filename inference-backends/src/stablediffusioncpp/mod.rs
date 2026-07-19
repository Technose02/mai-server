mod stablediffusionconfig;
pub use stablediffusionconfig::{StableDiffusionCppConfig, StableDiffusionEvent};

mod stablediffusionjob;
pub use stablediffusionjob::{SamplingMethod, Scheduler, StableDiffusionJob};

mod krea2turbo;
pub use krea2turbo::Krea2TurboJob;
mod animaturbo;
pub use animaturbo::AnimaTurboJob;
mod zimage;
pub use zimage::ZImageJob;
mod zimageturbo;
pub use zimageturbo::ZImageTurboJob;

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
