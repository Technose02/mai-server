use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, HasBaseJob, JobBase},
};

pub struct Flux2Klein9bJob(JobBase);

impl Default for Flux2Klein9bJob {
    fn default() -> Self {
        Self(JobBase {
            path_to_model: "/model_data/huggingface/unsloth/FLUX.2-klein-9B-GGUF/flux-2-klein-9b-BF16.gguf".into(),
            textencoder: ClipModel::llm("/model_data/huggingface/Comfy-Org/flux2-klein-9B/split_files/text_encoders/qwen_3_8b.safetensors"),
            path_to_vae: "/model_data/huggingface/Comfy-Org/flux2-klein-4B/split_files/vae/flux2-vae.safetensors".into(),
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            guidance: 3.5,
            prompt: "A Logo in white on black background saying 'Flux2 Klein 9B' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        })
    }
}

impl HasBaseJob for Flux2Klein9bJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for Flux2Klein9bJob {}
