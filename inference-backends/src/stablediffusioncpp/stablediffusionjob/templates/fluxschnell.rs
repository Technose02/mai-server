use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn flux_schnell_job() -> Self {
        let mut job = StableDiffusionJob::default();
        job.path_to_model = "/model_data/comfyui-model-base/unet/flux1-schnell-Q8_0.gguf".into();
        job.path_to_vae = "/model_data/comfyui-model-base/vae/flux-vae.safetensors".into();
        job.textencoder = ClipModel::clipl_and_t5xxl(
            "/model_data/comfyui-model-base/clip/clip_l.safetensors",
            "/model_data/comfyui-model-base/clip/t5xxl_fp8_e4m3fn.safetensors",
        );
        job.steps = 4;
        job.width = 1024;
        job.height = 1024;
        job.cfg_scale = 1.0;
        job.offload_to_cpu = false;
        job.flash_attention_mode = FlashAttentionMode::Full;
        job.prompt= "A Logo in white on black background saying 'Flux Schnell' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into();
        job
    }
}
