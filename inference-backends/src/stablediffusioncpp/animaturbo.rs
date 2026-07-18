use super::{SamplingMethod, Scheduler, StableDiffusionJob};
use std::path::{Path, PathBuf};

pub struct AnimaTurboJob {
    path_to_model: PathBuf,
    path_to_textencoder: PathBuf,
    path_to_vae: PathBuf,
    prompt: String,
    width: usize,
    height: usize,
    cfg_scale: f32,
    guidance: f32,
    vae_tiling: bool,
    offload_to_cpu: bool,
    seed: Option<u32>,
    steps: usize,
    scheduler: Scheduler,
    sampling_method: SamplingMethod,
}

impl Default for AnimaTurboJob {
    fn default() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/circlestone-labs/Anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors".into(),
            path_to_textencoder: "/model_data/huggingface/circlestone-labs/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors".into(),
            path_to_vae: "/model_data/huggingface/circlestone-labs/Anima/split_files/vae/qwen_image_vae.safetensors".into(),
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            vae_tiling: false,
            offload_to_cpu: false,
            seed: None,
            scheduler: Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            prompt: "A Logo in white on black background saying 'AnimaPreviewTurbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into()
        }
    }
}

impl StableDiffusionJob for AnimaTurboJob {
    fn diffusion_model(&self) -> &Path {
        &self.path_to_model
    }

    fn llm(&self) -> &Path {
        &self.path_to_textencoder
    }

    fn vae(&self) -> &Path {
        &self.path_to_vae
    }

    fn steps(&self) -> usize {
        self.steps
    }
    fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    fn prompt(&self) -> &str {
        &self.prompt
    }
    fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    fn width(&self) -> usize {
        self.width
    }
    fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    fn height(&self) -> usize {
        self.height
    }
    fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    fn cfg_scale(&self) -> f32 {
        self.cfg_scale
    }
    fn with_cfg_scale(mut self, cfg_scale: f32) -> Self {
        self.cfg_scale = cfg_scale;
        self
    }

    fn guidance(&self) -> f32 {
        self.guidance
    }
    fn with_guidance(mut self, guidance: f32) -> Self {
        self.guidance = guidance;
        self
    }

    fn vae_tiling(&self) -> bool {
        self.vae_tiling
    }
    fn with_vae_tiling(mut self, vae_tiling: bool) -> Self {
        self.vae_tiling = vae_tiling;
        self
    }

    fn offload_to_cpu(&self) -> bool {
        self.offload_to_cpu
    }

    fn with_offload_to_cpu(mut self, offload_to_cpu: bool) -> Self {
        self.offload_to_cpu = offload_to_cpu;
        self
    }

    fn seed(&self) -> Option<u32> {
        self.seed
    }
    fn with_seed(mut self, seed: u32) -> Self {
        self.seed = Some(seed);
        self
    }

    fn scheduler(&self) -> Scheduler {
        self.scheduler
    }
    fn with_scheduler(mut self, scheduler: Scheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    fn sampling_method(&self) -> SamplingMethod {
        self.sampling_method
    }

    fn with_sampling_method(mut self, sampling_method: SamplingMethod) -> Self {
        self.sampling_method = sampling_method;
        self
    }
}
