use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub enum VisionEncoder {
    #[default]
    None,
    LlmVision(PathBuf),
}

impl VisionEncoder {
    pub fn llm_vision(path: impl Into<PathBuf>) -> Self {
        Self::LlmVision(path.into())
    }
}
