use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn firered_image_edit_8steps_job() -> Self {
        let job = Self {
            path_to_model: "/model_data/huggingface/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/FireRed-Image-Edit-1.1-transformer.safetensors".into(),
            textencoder: ClipModel::llm("/model_data/comfyui-model-base/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            path_to_vae: "/model_data/comfyui-model-base/vae/firered-image-edit-1.1-vae.safetensors".into(),            
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Boogu Image Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        };
        job.with_lora("/model_data/huggingface/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors", 1.0)
        .expect("failed to add FireRed-Image-Edit-1.1-Lightning-8steps-v1.2")
    }
}
