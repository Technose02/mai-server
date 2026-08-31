use crate::stablediffusioncpp::{
    FlashAttentionMode, SamplingMethod, Scheduler, StableDiffusionJob,
    stablediffusionjob::{ClipModel, RefImageArgs},
};
use std::{collections::HashMap, path::PathBuf};

#[derive(Default)]
pub struct JobBase {
    pub path_to_model: PathBuf,
    pub textencoder: ClipModel,
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
    pub ref_image_args: Option<RefImageArgs>,
    pub init_image: Option<Vec<u8>>,
    pub ref_image_1: Option<Vec<u8>>,
    pub ref_image_2: Option<Vec<u8>>,
    pub ref_image_3: Option<Vec<u8>>,
    pub lora_models: HashMap<PathBuf, f32>,
}

pub trait HasBaseJob: Default {
    fn base(&self) -> &JobBase;
    fn base_mut(&mut self) -> &mut JobBase;
}

impl HasBaseJob for JobBase {
    fn base(&self) -> &JobBase {
        self
    }
    fn base_mut(&mut self) -> &mut JobBase {
        self
    }
}

impl StableDiffusionJob for JobBase {}
