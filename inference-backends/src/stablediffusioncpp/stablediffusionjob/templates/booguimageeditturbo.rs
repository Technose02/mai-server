use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder, VisionEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn boogu_image_edit_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/Comfy-Org/Boogu-Image/diffusion_models/boogu_image_edit_turbo_hotfix_1k_20260708_bf16.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/comfyui-model-base/text_encoders/Qwen3VL-8B-Instruct-Q8_0.gguf"),
            vision_encoder: VisionEncoder::llm_vision("/model_data/comfyui-model-base/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
            path_to_vae: "/model_data/huggingface/Comfy-Org/Boogu-Image/vae/flux1_vae_bf16.safetensors".into(),            
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Simple,
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            guidance: 0.0,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            prompt: "A Logo in white on black background saying 'Boogu Image Edit Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
