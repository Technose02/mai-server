use std::path::PathBuf;

#[derive(Clone, Debug, Default)]
pub enum ClipModel {
    #[default]
    None,
    Llm(PathBuf),
    CliplAndT5XXL {
        clip_l: PathBuf,
        t5xxl: PathBuf,
    },
}

impl ClipModel {
    pub fn llm(path: impl Into<PathBuf>) -> Self {
        Self::Llm(path.into())
    }
    pub fn clipl_and_t5xxl(clip_l_path: impl Into<PathBuf>, t5xx_path: impl Into<PathBuf>) -> Self {
        Self::CliplAndT5XXL {
            clip_l: clip_l_path.into(),
            t5xxl: t5xx_path.into(),
        }
    }
}
