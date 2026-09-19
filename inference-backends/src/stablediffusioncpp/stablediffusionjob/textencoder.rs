use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub enum TextEncoder {
    #[default]
    None,
    Llm(PathBuf),
    T5XXL(PathBuf),
    CliplAndT5XXL {
        clip_l: PathBuf,
        t5xxl: PathBuf,
    },
}

impl TextEncoder {
    pub fn llm(path: impl Into<PathBuf>) -> Self {
        Self::Llm(path.into())
    }
    pub fn t5xxl(path: impl Into<PathBuf>) -> Self {
        Self::T5XXL(path.into())
    }
    pub fn clipl_and_t5xxl(clip_l_path: impl Into<PathBuf>, t5xx_path: impl Into<PathBuf>) -> Self {
        Self::CliplAndT5XXL {
            clip_l: clip_l_path.into(),
            t5xxl: t5xx_path.into(),
        }
    }
}
