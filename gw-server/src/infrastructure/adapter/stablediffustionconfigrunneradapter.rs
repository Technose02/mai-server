use crate::domain::ports::StableDiffusionConfigRunnerOutPort;
use async_trait::async_trait;
use axum::http::StatusCode;
use inference_backends::stablediffusioncpp::StableDiffusionCppConfig;
use inference_backends::stablediffusioncpp::{StableDiffusionEvent, StableDiffusionJob};
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{Mutex, mpsc::Receiver};
use tracing::error;

pub struct StableDiffusionConfigRunnerAdapter {
    path_to_sd_exe: PathBuf,
    config: Arc<Mutex<Option<StableDiffusionCppConfig>>>,
}

impl StableDiffusionConfigRunnerAdapter {
    pub fn create_adapter(
        path_to_sd_exe: impl Into<PathBuf>,
    ) -> Arc<dyn StableDiffusionConfigRunnerOutPort> {
        Arc::new(StableDiffusionConfigRunnerAdapter {
            path_to_sd_exe: path_to_sd_exe.into(),
            config: Arc::new(Mutex::new(None)),
        })
    }
}

#[async_trait]
impl StableDiffusionConfigRunnerOutPort for StableDiffusionConfigRunnerAdapter {
    async fn abort_all(&self) {
        if let Some(config) = &mut *self.config.lock().await {
            config.stop().await;
        }
    }

    async fn run_job(
        &self,
        job: StableDiffusionJob,
    ) -> Result<Receiver<StableDiffusionEvent>, axum::http::StatusCode> {
        self.abort_all().await;

        let mut sdcfg =
            StableDiffusionCppConfig::init_with_temp_dir(self.path_to_sd_exe.clone(), "/tmp")
                .map_err(|e| {
                    error!("error creating StableDiffusionCppConfig: {e}");
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

        let res = sdcfg.run(&job).map_err(|e| {
            error!("error running z_image_turbo_job: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        });

        *self.config.lock().await = Some(sdcfg);

        res
    }
}
