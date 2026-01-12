use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Llmodels {
    models: Vec<InternalModel>,
    object: String,
    data: Vec<InternalData>,
}

impl Llmodels {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            object: "list".into(),
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, model: Model, data: Data) {
        self.models.push(model.into());
        self.data.push(InternalData {
            id: data.id,
            object: "model".into(),
            created: data.created,
            owned_by: "Mai-Server".into(),
            meta: data.meta,
        });
    }
}

#[derive(Debug, Serialize)]
struct InternalModel {
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

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: String,
    pub digest: String,
    pub description: String,
    pub tags: Vec<String>,
    pub capabilities: Vec<String>,
    pub parameters: String,
    pub details: ModelDetails,
}

impl From<Model> for InternalModel {
    fn from(creator: Model) -> Self {
        Self {
            name: creator.name,
            model: creator.model,
            modified_at: creator.modified_at,
            size: creator.size,
            digest: creator.digest,
            r#type: "model".into(),
            description: creator.description,
            tags: creator.tags,
            capabilities: creator.capabilities,
            parameters: creator.parameters,
            details: creator.details,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: Vec<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize)]
struct InternalData {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
    meta: DataMeta,
}

#[derive(Debug, Serialize)]
pub struct Data {
    pub id: String,
    pub created: u64,
    pub meta: DataMeta,
}

#[derive(Debug, Serialize)]
pub struct DataMeta {
    pub vocab_type: u8,
    pub n_vocab: u64,
    pub n_ctx_train: u64,
    pub n_embd: u64,
    pub n_params: u64,
    pub size: u64,
}
