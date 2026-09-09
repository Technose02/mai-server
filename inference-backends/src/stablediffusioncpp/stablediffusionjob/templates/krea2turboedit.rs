use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode, RefImageArgs},
};

impl StableDiffusionJob {
    pub fn krea2_turbo_edit_job() -> Self {
        let job = Self {
            path_to_model: "/model_data/huggingface/Comfy-Org/Krea-2/diffusion_models/krea2_turbo_bf16.safetensors".into(),
            path_to_vae: "/model_data/huggingface/Comfy-Org/Krea-2/vae/qwen_image_vae.safetensors".into(),
            textencoder: ClipModel::llm("/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf"),
            steps: 8,
            width: 1024,
            height: 1024,
            cfg_scale: 7.0,
            guidance: 3.5,
            offload_to_cpu: false,
            flash_attention_mode: FlashAttentionMode::Full,
            prompt: "A Logo in white on black background saying 'Krea2 Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        };

        job.with_ref_image_args(RefImageArgs::PresetKrea2Edit)
        .with_lora("/model_data/huggingface/conradlocke/krea2-identity-edit/krea2_identity_edit_v1_2.safetensors", 1.0)
        .expect("failed to add krea2_identity_edit_v1_2")
    }
}
