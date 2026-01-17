use crate::domain::ports::{
    ModelManagerServiceInPort, ModelsServiceInPort, OpenAiRequestForwardPServiceInPort,
};
use std::{borrow::Cow, sync::Arc};

pub trait SecurityConfig: Send + Sync + 'static {
    fn get_apikey(&self) -> Cow<'_, str>;
}

pub(crate) trait ApplicationConfig: Send + Sync + 'static {
    fn openai_service(&self) -> Arc<dyn OpenAiRequestForwardPServiceInPort>;
    fn modelmanager_service(&self) -> Arc<dyn ModelManagerServiceInPort>;
    fn models_service(&self) -> Arc<dyn ModelsServiceInPort>;
    fn increase_number_of_parallel_requests(&self) -> bool;
    fn decrease_increase_number_of_parallel_requests(&self);
}
