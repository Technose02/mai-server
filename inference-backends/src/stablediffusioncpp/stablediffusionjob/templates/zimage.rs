use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn z_image_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/Comfy-Org/z_image/split_files/diffusion_models/z_image_bf16.safetensors".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/z_image/split_files/vae/ae.safetensors".into(),
            textencoder: ClipModel::llm("/model_data/huggingface/Comfy-Org/z_image/split_files/text_encoders/qwen_3_4b.safetensors"),
            steps: 28,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Z Image' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
