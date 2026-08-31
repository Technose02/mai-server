use crate::{
    application::model::{StableDiffusionPromptDto, StableDiffusionSse},
    domain::ports::{StableDiffusionConfigRunnerOutPort, StableDiffusionServiceInPort},
};
use async_stream::stream;
use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use base64::prelude::{BASE64_STANDARD, Engine};
use inference_backends::stablediffusioncpp::StableDiffusionEvent;
use std::sync::Arc;

pub struct DefaultStableDiffusionService {
    stable_diffusion_config_runner: Arc<dyn StableDiffusionConfigRunnerOutPort>,
}

impl DefaultStableDiffusionService {
    pub fn create_service(
        stable_diffusion_config_runner: Arc<dyn StableDiffusionConfigRunnerOutPort>,
    ) -> Arc<dyn StableDiffusionServiceInPort> {
        Arc::new(Self {
            stable_diffusion_config_runner,
        })
    }
}

#[async_trait]
impl StableDiffusionServiceInPort for DefaultStableDiffusionService {
    async fn abort_all(&self) {
        self.stable_diffusion_config_runner.abort_all().await;
    }

    async fn process_prompt(
        &self,
        sd_config: &str,
        prompt_dto: StableDiffusionPromptDto,
    ) -> Result<Response, StatusCode> {
        let mut receiver = self
            .stable_diffusion_config_runner
            .create_and_run_job(sd_config, prompt_dto)
            .await?;

        let stream = stream! {

            let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(10));
            let mut progress = None;

            loop {

                tokio::select! {

                    _ = heartbeat_interval.tick() => {
                        let json = if let Some((step,nsteps)) = progress {
                            serde_json::to_string(&StableDiffusionSse::Progress{step, nsteps}).unwrap()
                        } else {
                            serde_json::to_string(&&StableDiffusionSse::GenerationStarted).unwrap()
                        };
                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                    }
                    optional_evt = receiver.recv() => {
                        if let Some(evt) = optional_evt {
                            heartbeat_interval.reset();

                            match evt {
                                StableDiffusionEvent::GenerationStarted { seed: _, started_at: _ } => {
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::GenerationStarted) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::GenerationStarted"))
                                    }
                                    continue;
                                },
                                StableDiffusionEvent::Progress {step,nsteps, duration: _} => {
                                    progress = Some((step, nsteps));
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::Progress {step, nsteps}) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::Progress"))
                                    }
                                    continue;
                                },
                                StableDiffusionEvent::StdOutLine(text) => {
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::StdOutLine{text}) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::StdOutLine"))
                                    }
                                    continue;
                                },
                                StableDiffusionEvent::StdErrLine(text) => {
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::StdErrLine{text}) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::StdErrLine"))
                                    }
                                    continue;
                                },
                                StableDiffusionEvent::Error(e) => {
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::Error{message: e.to_string()}) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::Error"))
                                    }
                                    break;
                                },
                                StableDiffusionEvent::GenerationFinished{boxed_data, duration:_} => {
                                    let b64_encoded_image = BASE64_STANDARD.encode(*boxed_data);
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::GenerationFinished{b64_encoded_image}) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::GenerationFinished"))
                                    }
                                    break;
                                },
                                StableDiffusionEvent::Killed => {
                                    if let Ok(json) = serde_json::to_string(&StableDiffusionSse::Killed) {
                                        yield Ok::<_, std::convert::Infallible>(Event::default().data(json))
                                    } else {
                                        yield Ok(Event::default().event("error").data("error serializing StableDiffusionSse::Killed"))
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        };
        Ok(Sse::new(stream).into_response())
    }
}
