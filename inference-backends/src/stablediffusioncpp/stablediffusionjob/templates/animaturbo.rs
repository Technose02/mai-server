use crate::stablediffusioncpp::{
    SamplingMethod, Scheduler, StableDiffusionJob,
    stablediffusionjob::{HasBaseJob, JobBase},
};

pub struct AnimaTurboJob(JobBase);

impl Default for AnimaTurboJob {
    fn default() -> Self {
        Self(JobBase {
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
            prompt: "A Logo in white on black background saying 'Anima Preview Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into()
        })
    }
}

impl HasBaseJob for AnimaTurboJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for AnimaTurboJob {}
