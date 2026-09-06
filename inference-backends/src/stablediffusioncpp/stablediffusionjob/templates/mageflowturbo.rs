use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn mage_flow_turbo_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/transformer/diffusion_pytorch_model.safetensors".into(),
            path_to_vae: "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/vae/diffusion_pytorch_model.safetensors".into(),
            textencoder: ClipModel::llm("/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf"),
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            guidance: 1.0,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Mage-Flow Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
