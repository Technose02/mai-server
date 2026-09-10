use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn boogu_image_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/Comfy-Org/Boogu-Image/diffusion_models/boogu_image_turbo_bf16.safetensors".into(),
            textencoder: ClipModel::llm("/model_data/huggingface/Comfy-Org/Ideogram-4/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
            path_to_vae: "/model_data/huggingface/Comfy-Org/Boogu-Image/vae/flux1_vae_bf16.safetensors".into(),            
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Boogu Image Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
