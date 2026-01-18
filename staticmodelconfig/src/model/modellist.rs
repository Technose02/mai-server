use super::modelconfiguration::ModelConfiguration;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

const OWNER: &str = "Mai-Server";

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelList {
    models: Vec<Model>,
    object: String,
    data: Vec<Data>,
}

impl Default for ModelList {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            object: "list".into(),
            data: Vec::new(),
        }
    }
}

impl ModelList {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            models: Vec::with_capacity(capacity),
            object: "list".into(),
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn as_list_of_meta_data_and_capabilities(&self) -> Vec<(DataMeta, Vec<String>)> {
        let mut ret = Vec::with_capacity(self.models.len());
        for (model, data) in self.models.iter().zip(self.data.iter()) {
            let meta_data = DataMeta {
                vocab_type: data.meta.vocab_type,
                n_vocab: data.meta.n_vocab,
                n_ctx_train: data.meta.n_ctx_train,
                n_embd: data.meta.n_embd,
                n_params: data.meta.n_params,
                size: data.meta.size,
            };
            let capabilities = model.capabilities.clone();
            ret.push((meta_data, capabilities));
        }
        ret
    }

    pub fn names(&self) -> String {
        self.models
            .iter()
            .map(|m| m.name.clone())
            .collect::<Vec<String>>()
            .join(",")
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

    pub fn add_model_configuration(&mut self, model_configuration: &ModelConfiguration) {
        let data_meta = DataMeta {
            vocab_type: model_configuration.vocab_type,
            n_vocab: model_configuration.n_vocab,
            n_embd: model_configuration.n_embd,
            n_ctx_train: model_configuration.n_ctx_train,
            n_params: model_configuration.n_params,
            size: model_configuration.size,
        };

        self.add_simple(
            model_configuration.alias.clone(),
            model_configuration.capabilities.clone(),
            data_meta,
        );
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails {
    parent_model: String,
    format: String,
    family: String,
    families: Vec<String>,
    parameter_size: String,
    quantization_level: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Data {
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

#[derive(Debug, Serialize, Deserialize)]
pub struct DataMeta {
    pub vocab_type: u8,
    pub n_vocab: u64,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_params: u64,
    pub size: u64,
}
