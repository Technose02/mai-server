use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder, VisionEncoder,
    },
};

impl StableDiffusionJob {
    pub fn qwen_image_21_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/qwen-image-2.1-Q4_K_M.gguf".into(),
            text_encoder: TextEncoder::llm("/model_data/comfyui-model-base/text_encoders/Qwen3VL-8B-Instruct-Q8_0.gguf"),
            path_to_vae: "/model_data/huggingface/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/vae/qwen_image_2.1_vae_bf16.safetensors".into(),
            vision_encoder: VisionEncoder::llm_vision("/model_data/comfyui-model-base/text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
            cfg_scale: 6.0,
            guidance: 1.0,
            scheduler: Scheduler::Simple,
            sampling_method: SamplingMethod::Euler,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            steps: 40,
            width: 2048,
            height: 2048,
            prompt: "A Logo in white on black background saying 'qwen2.1.cpp' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
