use crate::domain::ports::{
    ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
    StableDiffusionServiceInPort,
};
use std::{borrow::Cow, sync::Arc};

pub trait SecurityConfig: Send + Sync + 'static {
    fn get_apikey(&self) -> Option<Cow<'_, str>>;
}

pub trait ApplicationConfig: Send + Sync + 'static {
    fn openai_chat_completions_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn openai_embeddings_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn languagemodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn embeddingmodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn models_service(&self) -> Arc<dyn ModelsServiceInPort>;
    fn stable_diffusion_service(&self) -> Arc<dyn StableDiffusionServiceInPort>;

    /// when true the chat-ui document route does NOT try to get a language
    /// model served before forwarding (used by test/fixture instances that
    /// must never spawn real llama-server processes)
    fn skip_chatui_model_ensure(&self) -> bool {
        false
    }

    /// when true `/chat/sw.js` is forwarded to the backend, i.e. the upstream
    /// pwa service worker stays active. Defaults to false: the upstream worker
    /// precaches the document + bundle and pins a stale ui across backend
    /// rebuilds when the ui is served under the `/chat` prefix.
    fn serve_chatui_pwa_service_worker(&self) -> bool {
        false
    }
}
