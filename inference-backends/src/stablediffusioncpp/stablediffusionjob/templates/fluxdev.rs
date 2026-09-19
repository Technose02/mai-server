use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn flux_dev_job() -> Self {
        Self {
            path_to_model: "/model_data/comfyui-model-base/diffusion_models/flux1-dev.safetensors".into(),
            text_encoder: TextEncoder::clipl_and_t5xxl("/model_data/comfyui-model-base/clip/clip_l.safetensors",
            "/model_data/comfyui-model-base/clip/t5xxl_fp16.safetensors"),
            path_to_vae: "/model_data/comfyui-model-base/vae/flux-vae.safetensors".into(),
            cfg_scale: 1.0,
            offload_to_cpu: false,
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Simple,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            steps: 20,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'Flux Dev' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
