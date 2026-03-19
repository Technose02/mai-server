use crate::domain::ports::{
    ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
};
use std::{borrow::Cow, sync::Arc};

pub trait SecurityConfig: Send + Sync + 'static {
    fn get_apikey(&self) -> Option<Cow<'_, str>>;
}

pub(crate) trait ApplicationConfig: Send + Sync + 'static {
    fn openai_chat_completions_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn openai_embeddings_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn languagemodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn embeddingmodelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn models_service(&self) -> Arc<dyn ModelsServiceInPort>;
}
