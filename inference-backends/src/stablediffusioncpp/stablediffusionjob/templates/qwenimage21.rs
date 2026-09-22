use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder, VisionEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn qwen_image_21_job() -> Self {
        Self {
            //path_to_model: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/qwen-image-2.1-Q4_K_M.gguf".into(),
            path_to_model: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/qwen-image-2.1-Q8_0.gguf".into(),
            path_to_vae: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/vae/qwen_image_2.1_vae_bf16.safetensors".into(),
            //text_encoder: TextEncoder::llm("/model_data/huggingface/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"),
            text_encoder: TextEncoder::llm("/model_data/huggingface/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-BF16.gguf"),
            vision_encoder: VisionEncoder::llm_vision("/model_data/huggingface/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-mmproj-f16.gguf"),
            cfg_scale: 6.0,
            guidance: 1.0,
            scheduler: Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            offload_to_cpu: false,
            //flash_attention_mode: FlashAttentionMode::Full,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu).vae(BackendMode::Cpu),
            steps: 40,
            width: 2048,
            height: 2048,
            prompt: "A Logo in white on black background saying 'qwen2.1.cpp' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
