use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        FlashAttentionMode, TextEncoder, VisionEncoder,
    },
};

impl StableDiffusionJob {
    pub fn qwen_image_21_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/qwen-image-2.1-UC-BF16.gguf".into(),
            path_to_vae: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/vae/qwen_image_2.1_vae_bf16.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/huggingface/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-BF16.gguf"),
            vision_encoder: VisionEncoder::llm_vision("/model_data/huggingface/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive-mmproj-f16.gguf"),
            guidance: 1.0,
            steps: 5,
            cfg_scale: 1.0,
            sigmas: vec![1.0, 0.875, 0.75, 0.5, 0.25],
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'qwen2.1.cpp' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
        .with_lora("/model_data/comfyui-model-base/loras/Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors", 1.0)
        .expect("failed to apply lora 'Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256'")

    }
}
