use crate::stablediffusioncpp::{
    StableDiffusionJob,
    stablediffusionjob::{ClipModel, HasBaseJob, JobBase},
};

pub struct FluxSchnellJob(JobBase);

impl Default for FluxSchnellJob {
    fn default() -> Self {
        Self(JobBase {
            path_to_model: "/model_data/comfyui-model-base/unet/flux1-schnell-Q8_0.gguf".into(),
            path_to_vae: "/model_data/comfyui-model-base/vae/flux-vae.safetensors".into(),
            textencoder: ClipModel::clipl_and_t5xxl("/model_data/comfyui-model-base/clip/clip_l.safetensors", "/model_data/comfyui-model-base/clip/t5xxl_fp8_e4m3fn.safetensors"),
            steps: 4,
            width: 1024,
            height: 1024,
            cfg_scale: 1.0,
            prompt: "A Logo in white on black background saying 'Flux Schnell' in capitals using a classic computer terminal font. Text is centered horizontally and vertically".into(),
            ..Default::default()
        })
    }
}

impl HasBaseJob for FluxSchnellJob {
    fn base(&self) -> &JobBase {
        &self.0
    }

    fn base_mut(&mut self) -> &mut JobBase {
        &mut self.0
    }
}

impl StableDiffusionJob for FluxSchnellJob {}
