use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn anima_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/circlestone-labs/Anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors".into(),
            path_to_vae: "/model_data/huggingface/circlestone-labs/Anima/split_files/vae/qwen_image_vae.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/huggingface/circlestone-labs/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors"),
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            scheduler: Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            guidance: 0.0,
            prompt: "A Logo in white on black background saying 'Anima Preview Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
