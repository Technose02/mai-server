use crate::stablediffusioncpp::{FlashAttentionMode, SamplingMethod, Scheduler};
use std::path::PathBuf;

pub struct JobBase {
    pub path_to_model: PathBuf,
    pub path_to_textencoder: PathBuf,
    pub path_to_vae: PathBuf,
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub cfg_scale: f32,
    pub guidance: f32,
    pub vae_tiling: bool,
    pub offload_to_cpu: bool,
    pub flash_attention_mode: FlashAttentionMode,
    pub seed: Option<u32>,
    pub steps: usize,
    pub scheduler: Scheduler,
    pub sampling_method: SamplingMethod,
    pub ref_image_1: Option<Vec<u8>>,
    pub ref_image_2: Option<Vec<u8>>,
    pub ref_image_3: Option<Vec<u8>>,
}

pub trait HasBaseJob: Default {
    fn base(&self) -> &JobBase;
    fn base_mut(&mut self) -> &mut JobBase;
}
