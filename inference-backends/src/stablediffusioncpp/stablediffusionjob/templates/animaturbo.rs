use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn anima_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/circlestone-labs/Anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors".into(),
            path_to_vae: "/model_data/huggingface/circlestone-labs/Anima/split_files/vae/qwen_image_vae.safetensors".into(),
            textencoder: ClipModel::llm("/model_data/huggingface/circlestone-labs/Anima/split_files/text_encoders/qwen_3_06b_base.safetensors"),
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Anima Preview Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
