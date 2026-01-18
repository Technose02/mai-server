use crate::{SecurityConfig, domain::ports::ModelLoaderOutPort};
use async_trait::async_trait;
use inference_backends::LlamaCppConfigArgs;
use staticmodelconfig::{ContextSizeAwareAlias, ModelConfiguration};
use std::{error::Error, path::Path, sync::Arc};

pub struct StaticModelLoader {
    static_model_configuration_list: Vec<ModelConfiguration>,
    security_config: Arc<dyn SecurityConfig>,
}

impl StaticModelLoader {
    pub fn create_adapter(
        configurations_dir: &Path,
        security_config: Arc<dyn SecurityConfig>,
    ) -> Result<Arc<dyn ModelLoaderOutPort>, Box<dyn Error>> {
        Ok(Arc::new(Self {
            static_model_configuration_list: ModelConfiguration::load_from_json_files(
                configurations_dir,
            )?
            .0,
            security_config,
        }))
    }
}

#[async_trait]
impl ModelLoaderOutPort for StaticModelLoader {
    fn get_static_model_configurations(&self) -> Vec<ModelConfiguration> {
        self.static_model_configuration_list.clone()
    }

    async fn get_model_configuration(&self, alias: &str) -> Result<Arc<LlamaCppConfigArgs>, ()> {
        let alias = alias.to_owned();
        let (model_key, optional_context_size) =
            match ContextSizeAwareAlias::try_from(alias.clone()) {
                Ok(caa) => (caa.model(), Some(caa.context_size())),
                Err(_) => (alias.clone(), None),
            };

        if let Some(model_configuration) = self
            .static_model_configuration_list
            .iter()
            .find(|&config| config.alias == model_key)
        {
            Ok(Arc::new(LlamaCppConfigArgs {
                alias,
                api_key: Some(self.security_config.get_apikey().to_string()),
                model_path: model_configuration.model_path.clone(),
                mmproj_path: model_configuration.mmproj_path.clone(),
                prio: model_configuration.prio,
                threads: model_configuration.threads,
                n_gpu_layers: model_configuration.n_gpu_layers,
                jinja: model_configuration.jinja,
                ctx_size: optional_context_size,
                no_mmap: model_configuration.no_mmap,
                flash_attn: model_configuration.flash_attn.clone(),
                fit: model_configuration.fit.clone(),
                batch_size: model_configuration.batch_size,
                ubatch_size: model_configuration.ubatch_size,
                cache_type_k: model_configuration.cache_type_k.clone(),
                cache_type_v: model_configuration.cache_type_v.clone(),
                no_context_shift: model_configuration.no_context_shift,
                no_cont_batching: model_configuration.no_cont_batching,
                min_p: model_configuration.min_p,
                temp: model_configuration.temp,
                repeat_penalty: model_configuration.repeat_penalty,
                presence_penalty: model_configuration.presence_penalty,
                seed: model_configuration.seed,
                top_k: model_configuration.top_k,
                top_p: model_configuration.top_p,
            }))
        } else {
            eprintln!("no model-configuration found for alias '{model_key}'");
            Err(())
        }
    }
}
