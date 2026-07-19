use crate::stablediffusioncpp::{
    SamplingMethod, Scheduler, StableDiffusionJob,
    stablediffusionjob::{HasBaseJob, JobBase},
};

pub struct ZImageJob(JobBase);

impl Default for ZImageJob {
    fn default() -> Self {
        Self(JobBase {
            path_to_model: "/model_data/huggingface/Comfy-Org/z_image/split_files/diffusion_models/z_image_bf16.safetensors".into(),
            path_to_textencoder: "/model_data/huggingface/Comfy-Org/z_image/split_files/text_encoders/qwen_3_4b.safetensors".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/z_image/split_files/vae/ae.safetensors".into(),
            steps: 28,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            vae_tiling: false,
            offload_to_cpu: true,
            seed: None,
            scheduler: Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            prompt: "A Logo in white on black background saying 'Z-Image' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into()
        })
    }
}

impl HasBaseJob for ZImageJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for ZImageJob {}
