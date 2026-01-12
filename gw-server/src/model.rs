use crate::domain::ports::{
    ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
};
use std::{borrow::Cow, sync::Arc};

pub trait SecurityConfig: Send + Sync + 'static {
    fn get_apikey(&self) -> Cow<'_, str>;
}

pub(crate) trait ApplicationConfig: Send + Sync + 'static {
    fn openapi_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn modelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn models_service(&self) -> Arc<dyn ModelsServiceInPort>;
}
