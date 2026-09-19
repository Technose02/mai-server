use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder, VisionEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn mage_flow_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/transformer/diffusion_pytorch_model.safetensors".into(),
            path_to_vae: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/vae/diffusion_pytorch_model.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/comfyui-model-base/text_encoders/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf"),
            vision_encoder: VisionEncoder::llm_vision("/model_data/comfyui-model-base/text_encoders/Qwen3-VL-4B-Instruct-Uncensored.mmproj-f16.gguf"),
            cfg_scale: 1.0,
            guidance: 1.0,
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Simple,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            steps: 4,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'Mage-Flow Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
