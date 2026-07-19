use crate::stablediffusioncpp::{
    SamplingMethod, Scheduler, StableDiffusionJob,
    stablediffusionjob::{HasBaseJob, JobBase},
};

pub struct Krea2TurboJob(JobBase);

impl Default for Krea2TurboJob {
    fn default() -> Self {
        Self(JobBase {
            path_to_model: "/model_data/huggingface/Comfy-Org/Krea-2/diffusion_models/krea2_turbo_bf16.safetensors".into(),
            path_to_textencoder: "/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/Krea-2/vae/qwen_image_vae.safetensors".into(),
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            vae_tiling: false,
            offload_to_cpu: false,
            seed: None,
            scheduler:Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            prompt: "A Logo in white on black background saying 'Krea2 Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into()
        })
    }
}

impl HasBaseJob for Krea2TurboJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for Krea2TurboJob {}
