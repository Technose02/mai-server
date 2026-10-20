use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{FlashAttentionMode, SamplingMethod, Scheduler, TextEncoder},
};

impl StableDiffusionJob {
    pub fn moss_sft_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/axiomofmind/Moss-Image-SFT-GGUF/MOSS-Image-SFT-Q8_0.gguf".into(),
            path_to_vae: "/model_data/huggingface/OpenMOSS-Team/MOSS-Image-SFT/vae/diffusion_pytorch_model.safetensors".into(),
            text_encoder: TextEncoder::llm("/model_data/huggingface/llmfan46/Qwen3.5-9B-Ultra-Uncensored-Heretic-GGUF/Qwen3.5-9B-Ultra-Uncensored-Heretic-Q8_0.gguf"),
            path_to_tokenizer: Some("/model_data/huggingface/OpenMOSS-Team/MOSS-Image-SFT/tokenizer/tokenizer.json".into()),
            cfg_scale: 3.0,
            steps: 50,
            sampling_method: SamplingMethod::Euler,
            scheduler: Scheduler::Discrete,
            flow_shift: Some(5.0),
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            width: 1024,
            height: 1024,
            prompt: "A Logo in white on black background saying 'MOSS-SFT' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
