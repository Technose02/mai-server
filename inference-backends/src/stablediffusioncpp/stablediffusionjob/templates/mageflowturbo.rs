use crate::stablediffusioncpp::{
    SamplingMethod, Scheduler, StableDiffusionJob,
    stablediffusionjob::{FlashAttentionMode, HasBaseJob, JobBase},
};

pub struct MageFlowTurboJob(JobBase);

impl Default for MageFlowTurboJob {
    fn default() -> Self {
        Self(JobBase {
            path_to_model: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/transformer/diffusion_pytorch_model.safetensors".into(),
            path_to_textencoder: "/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf".into(),
            path_to_vae: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/vae/diffusion_pytorch_model.safetensors".into(),
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            vae_tiling: false,
            flash_attention_mode: FlashAttentionMode::None,
            offload_to_cpu: false,
            seed: None,
            guidance: 1.0,
            scheduler:Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            ref_image_1: None,
            ref_image_2: None,
            ref_image_3: None,
            prompt: "A Logo in white on black background saying 'Mage-Flow Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into()
        })
    }
}

impl HasBaseJob for MageFlowTurboJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for MageFlowTurboJob {}
