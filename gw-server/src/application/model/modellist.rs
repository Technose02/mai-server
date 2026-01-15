use crate::domain::model::ModelConfiguration as DomainModelConfiguration;
use inference_backends::ContextSize;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

const OWNER: &str = "Mai-Server";

#[derive(Debug, Serialize)]
pub struct ModelList {
    models: Vec<Model>,
    object: String,
    data: Vec<Data>,
}

impl ModelList {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            object: "list".into(),
            data: Vec::new(),
        }
    }

    fn add_simple(
        &mut self,
        alias: impl Into<String>,
        capabilities: Vec<String>,
        data_meta: DataMeta,
    ) {
        let alias = alias.into();
        self.models.push(Model::simple(alias.clone(), capabilities));
        self.data.push(Data::simple(alias, data_meta));
    }

    pub fn extend_from_domain_model_configuration(
        &mut self,
        domain_model_configuration: &DomainModelConfiguration,
    ) {
        for ctx_size in [
            ContextSize::T8192,
            ContextSize::T16384,
            ContextSize::T32768,
            ContextSize::T65536,
            ContextSize::T131072,
            ContextSize::T262144,
        ] {
            if ctx_size > domain_model_configuration.max_ctx_size {
                break;
            }

            self.add_domain_model_configuration(domain_model_configuration);
        }
    }

    fn add_domain_model_configuration(
        &mut self,
        domain_model_configuration: &DomainModelConfiguration,
    ) {
        let data_meta = DataMeta {
            vocab_type: domain_model_configuration.vocab_type,
            n_vocab: domain_model_configuration.n_vocab,
            n_embd: domain_model_configuration.n_embd,
            n_ctx_train: domain_model_configuration.n_ctx_train,
            n_params: domain_model_configuration.n_params,
            size: domain_model_configuration.size,
        };

        self.add_simple(
            domain_model_configuration.alias.clone(),
            domain_model_configuration.capabilities.clone(),
            data_meta,
        );
    }
}

#[derive(Debug, Serialize)]
struct Model {
    name: String,
    model: String,
    modified_at: String,
    size: String,
    digest: String,
    r#type: String,
    description: String,
    tags: Vec<String>,
    capabilities: Vec<String>,
    parameters: String,
    details: ModelDetails,
}

impl Model {
    fn simple(alias: impl Into<String>, capabilities: Vec<String>) -> Self {
        let name = alias.into();
        let model = name.clone();
        Self {
            name,
            model,
            modified_at: String::new(),
            size: String::new(),
            digest: String::new(),
            r#type: String::from("model"),
            description: String::new(),
            tags: vec![String::new()],
            capabilities,
            parameters: String::new(),
            details: ModelDetails {
                parent_model: String::new(),
                format: String::from("gguf"),
                family: String::new(),
                families: vec![String::new()],
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ModelDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize)]
struct Data {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
    meta: DataMeta,
}

impl Data {
    fn simple(alias: impl Into<String>, data_meta: DataMeta) -> Self {
        Self {
            id: alias.into(),
            object: String::from("model"),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("expected UNIX_EPOCH to be in the past")
                .as_secs(),
            owned_by: String::from(OWNER),
            meta: data_meta,
        }
    }
}

#[derive(Debug, Serialize)]
struct DataMeta {
    vocab_type: u8,
    n_vocab: u64,
    n_ctx_train: u64,
    n_embd: u64,
    n_params: u64,
    size: u64,
}
