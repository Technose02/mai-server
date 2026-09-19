use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{
        BackendRouting, FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder,
        backendrouting::BackendMode,
    },
};

impl StableDiffusionJob {
    pub fn krea2_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/Comfy-Org/Krea-2/diffusion_models/krea2_turbo_bf16.safetensors".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/Krea-2/vae/qwen_image_vae.safetensors".into(),
            text_encoder: TextEncoder::llm("/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf"),
            cfg_scale: 1.0,
            guidance: 3.5,
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Simple,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::DiffusionOnly,
            backend_routing: BackendRouting::default().te(BackendMode::Cpu),
            steps: 12,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'Krea2 Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
