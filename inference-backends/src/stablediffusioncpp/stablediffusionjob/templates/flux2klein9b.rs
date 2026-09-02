use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn flux2_klein_9b_job() -> Self {
        Self {
            path_to_model: "/model_data/huggingface/unsloth/FLUX.2-klein-9B-GGUF/flux-2-klein-9b-BF16.gguf".into(),
            textencoder: ClipModel::llm("/model_data/huggingface/Comfy-Org/flux2-klein-9B/split_files/text_encoders/qwen_3_8b.safetensors"),
            path_to_vae: "/model_data/huggingface/Comfy-Org/flux2-klein-4B/split_files/vae/flux2-vae.safetensors".into(),            
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            guidance: 3.5,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Flux2 Klein 9B' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        }
    }
}
