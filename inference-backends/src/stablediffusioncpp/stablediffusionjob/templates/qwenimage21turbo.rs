use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{FlashAttentionMode, TextEncoder, VisionEncoder,SamplingMethod},
};

impl StableDiffusionJob {
    pub fn qwen_image_21_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/Qwen-Image-2.1-Turbo-Q8_0.gguf".into(),
            path_to_vae: "/model_data/huggingface/KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF/vae/qwen_image_2.1_vae_bf16.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/huggingface/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf"),
            vision_encoder: VisionEncoder::llm_vision("/model_data/huggingface/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf"),
            cfg_scale: 1.0,
            sampling_method: SamplingMethod::Euler,
            steps: 8,
            sigmas: vec![1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0],
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'qwen2.1.cpp' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
