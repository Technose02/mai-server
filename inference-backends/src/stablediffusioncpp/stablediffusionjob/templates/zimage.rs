use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn z_image_job() -> Self {
        let mut job = StableDiffusionJob::default();
        job.path_to_model = "/model_data/huggingface/Comfy-Org/z_image/split_files/diffusion_models/z_image_bf16.safetensors".into();
        job.path_to_vae =
            "/model_data/huggingface/Comfy-Org/z_image/split_files/vae/ae.safetensors".into();
        job.textencoder = ClipModel::llm(
            "/model_data/huggingface/Comfy-Org/z_image/split_files/text_encoders/qwen_3_4b.safetensors",
        );
        job.steps = 28;
        job.width = 1024;
        job.height = 1024;
        job.cfg_scale = 7.0;
        job.guidance = 3.5;
        job.offload_to_cpu = false;
        job.flash_attention_mode = FlashAttentionMode::Full;
        job.prompt = "A Logo in white on black background saying 'Z Image' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into();
        job
    }
}
