use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, FlashAttentionMode},
};

impl StableDiffusionJob {
    pub fn mage_flow_turbo_job() -> Self {
        let mut job = StableDiffusionJob::default();

        job.path_to_model = "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/transformer/diffusion_pytorch_model.safetensors".into();
        job.path_to_vae = "/model_data/huggingface/mage-flow-community/Mage-Flow-Turbo/vae/diffusion_pytorch_model.safetensors".into();
        job.textencoder =
            ClipModel::llm("/home/technose02/Downloads/Qwen3-VL-4B-Instruct-Uncensored.Q8_0.gguf");
        job.steps = 4;
        job.width = 1024;
        job.height = 1024;
        job.cfg_scale = 1.0;
        job.guidance = 1.0;
        job.prompt = "A Logo in white on black background saying 'Mage-Flow Turbo' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into();
        job.offload_to_cpu = false;
        job.flash_attention_mode = FlashAttentionMode::Full;
        job
    }
}
