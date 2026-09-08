use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

mod flashattentionmode;
pub use flashattentionmode::FlashAttentionMode;
mod scheduler;
pub use scheduler::Scheduler;
mod samplingmethod;
pub use samplingmethod::SamplingMethod;
mod clipmodel;
pub use clipmodel::ClipModel;
mod refimageargs;
use crate::stablediffusioncpp::{StableDiffusionError, StableDiffusionResult};
pub use refimageargs::RefImageArgs;

pub mod templates;

#[derive(Debug, Default)]
pub struct StableDiffusionJob {
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
    pub init_png: Option<Vec<u8>>,
    pub ref_png_1: Option<Vec<u8>>,
    pub ref_png_2: Option<Vec<u8>>,
    pub ref_png_3: Option<Vec<u8>>,
    pub lora_models: HashMap<PathBuf, f32>,
}

impl StableDiffusionJob {
    pub fn diffusion_model(&self) -> &Path {
        &self.path_to_model
    }

    pub fn textencoder(&self) -> &ClipModel {
        &self.textencoder
    }

    pub fn vae(&self) -> &Path {
        &self.path_to_vae
    }

    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    pub fn height(&self) -> usize {
        self.height
    }
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    pub fn cfg_scale(&self) -> f32 {
        self.cfg_scale
    }
    pub fn with_cfg_scale(mut self, cfg_scale: f32) -> Self {
        self.cfg_scale = cfg_scale;
        self
    }

    pub fn guidance(&self) -> f32 {
        self.guidance
    }
    pub fn with_guidance(mut self, guidance: f32) -> Self {
        self.guidance = guidance;
        self
    }

    pub fn vae_tiling(&self) -> bool {
        self.vae_tiling
    }
    pub fn with_vae_tiling(mut self, vae_tiling: bool) -> Self {
        self.vae_tiling = vae_tiling;
        self
    }

    pub fn flash_attention_mode(&self) -> FlashAttentionMode {
        self.flash_attention_mode
    }
    pub fn with_flash_attention_mode(mut self, flash_attention_mode: FlashAttentionMode) -> Self {
        self.flash_attention_mode = flash_attention_mode;
        self
    }

    pub fn offload_to_cpu(&self) -> bool {
        self.offload_to_cpu
    }

    pub fn with_offload_to_cpu(mut self, offload_to_cpu: bool) -> Self {
        self.offload_to_cpu = offload_to_cpu;
        self
    }

    pub fn seed(&self) -> Option<u32> {
        self.seed
    }
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn scheduler(&self) -> Scheduler {
        self.scheduler
    }
    pub fn with_scheduler(mut self, scheduler: Scheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    pub fn sampling_method(&self) -> SamplingMethod {
        self.sampling_method
    }
    pub fn with_sampling_method(mut self, sampling_method: SamplingMethod) -> Self {
        self.sampling_method = sampling_method;
        self
    }

    pub fn ref_image_args(&self) -> &Option<RefImageArgs> {
        &self.ref_image_args
    }
    pub fn with_ref_image_args(mut self, ref_image_args: RefImageArgs) -> Self {
        self.ref_image_args = Some(ref_image_args);
        self
    }

    pub fn init_png(&self) -> &Option<Vec<u8>> {
        &self.init_png
    }
    pub fn with_init_png(mut self, init_png_data: Vec<u8>) -> Self {
        self.init_png = Some(init_png_data);
        self
    }

    pub fn ref_png_1(&self) -> &Option<Vec<u8>> {
        &self.ref_png_1
    }
    pub fn with_ref_png_1(mut self, ref_png_data: Vec<u8>) -> Self {
        self.ref_png_1 = Some(ref_png_data);
        self
    }

    pub fn ref_png_2(&self) -> &Option<Vec<u8>> {
        &self.ref_png_2
    }
    pub fn with_ref_png_2(mut self, ref_png_data: Vec<u8>) -> Self {
        self.ref_png_2 = Some(ref_png_data);
        self
    }

    pub fn ref_png_3(&self) -> &Option<Vec<u8>> {
        &self.ref_png_3
    }
    pub fn with_ref_png_3(mut self, ref_png_data: Vec<u8>) -> Self {
        self.ref_png_3 = Some(ref_png_data);
        self
    }

    pub fn lora_models(&self) -> &HashMap<PathBuf, f32> {
        &self.lora_models
    }
    pub fn with_lora(
        mut self,
        path: impl Into<PathBuf>,
        weight: f32,
    ) -> StableDiffusionResult<Self> {
        let path = path.into();
        match (path.is_file(), weight >= 0.0) {
            (false, _) => Err(StableDiffusionError::Custom(format!(
                "invalid lora-path: '{}'",
                path.to_string_lossy()
            ))),
            (true, false) => Err(StableDiffusionError::Custom(format!(
                "lora-weight must not be negative; got '{weight}'"
            ))),
            _ => {
                self.lora_models.insert(path, weight);
                Ok(self)
            }
        }
    }
}
