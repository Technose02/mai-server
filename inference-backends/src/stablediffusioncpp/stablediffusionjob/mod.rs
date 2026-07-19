use std::path::Path;

mod scheduler;
pub use scheduler::Scheduler;
mod samplingmethod;
pub use samplingmethod::SamplingMethod;

pub trait StableDiffusionJob {
    fn diffusion_model(&self) -> &Path;
    fn llm(&self) -> &Path;
    fn vae(&self) -> &Path;
    fn steps(&self) -> usize;
    fn with_steps(self, steps: usize) -> Self;
    fn prompt(&self) -> &str;
    fn with_prompt(self, prompt: impl Into<String>) -> Self;
    fn width(&self) -> usize;
    fn with_width(self, width: usize) -> Self;
    fn height(&self) -> usize;
    fn with_height(self, height: usize) -> Self;
    fn cfg_scale(&self) -> f32;
    fn guidance(&self) -> f32;
    fn with_guidance(self, guidance: f32) -> Self;
    fn with_cfg_scale(self, cfg_scale: f32) -> Self;
    fn vae_tiling(&self) -> bool;
    fn with_vae_tiling(self, vae_tiling: bool) -> Self;
    fn offload_to_cpu(&self) -> bool;
    fn with_offload_to_cpu(self, offload_to_cpu: bool) -> Self;
    fn seed(&self) -> Option<u32>;
    fn with_seed(self, seed: u32) -> Self;
    fn scheduler(&self) -> Scheduler;
    fn with_scheduler(self, scheduler: Scheduler) -> Self;
    fn sampling_method(&self) -> SamplingMethod;
    fn with_sampling_method(self, sampling_method: SamplingMethod) -> Self;
}
