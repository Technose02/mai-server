use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn z_image_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/comfyui-model-base/diffusion_models/z_image_turbo_bf16.safetensors".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/z_image/split_files/vae/ae.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/huggingface/Comfy-Org/z_image/split_files/text_encoders/qwen_3_4b.safetensors"),
            cfg_scale: 1.0,
            guidance: 3.5,
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Simple,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            steps: 8,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'Z Image Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
