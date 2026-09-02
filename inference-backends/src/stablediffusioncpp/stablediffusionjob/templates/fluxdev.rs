use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn flux_dev_job() -> Self {
        Self {
            path_to_model: "/model_data/comfyui-model-base/diffusion_models/flux1-dev.safetensors".into(),
            textencoder: ClipModel::clipl_and_t5xxl("/model_data/comfyui-model-base/clip/clip_l.safetensors",
            "/model_data/comfyui-model-base/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            path_to_vae: "/model_data/comfyui-model-base/vae/flux-vae.safetensors".into(),
            steps: 20,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Flux Dev' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
