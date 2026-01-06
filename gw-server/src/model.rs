use inference_backends::{LlamaCppBackend, LlamaCppBackendController};
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct AppState {
    llamacpp_controller: LlamaCppBackendController,
    api_key: String,
}

pub struct AppStateBuilder {
    llamacpp_backend: LlamaCppBackend,
    api_key: Option<String>,
}

impl AppState {
    pub fn builder(llamacpp_backend: LlamaCppBackend) -> AppStateBuilder {
        AppStateBuilder {
            llamacpp_backend,
            api_key: None,
        }
    }

    pub fn api_key(&self) -> &str {
        self.api_key.as_str()
    }

    pub fn llamacpp_controller(&self) -> &LlamaCppBackendController {
        &self.llamacpp_controller
    }
}

impl AppStateBuilder {
    pub fn with_api_key(&mut self, api_key: impl Into<String>) -> &mut Self {
        self.api_key = Some(api_key.into());
        self
    }

    fn random_api_key(len: usize) -> String {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

        let mut rng = rand::rng();

        (0..len)
            .map(|_| {
                let idx = rng.random_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    pub async fn build(self) -> AppState {
        let llamacpp_controller =
            LlamaCppBackendController::init_backend(self.llamacpp_backend).await;
        let api_key = self.api_key.unwrap_or(Self::random_api_key(25));

        AppState {
            llamacpp_controller,
            api_key,
        }
    }
}
