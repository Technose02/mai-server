mod comfyui;
mod llamacpp;
pub mod stablediffusioncpp;

pub use comfyui::{
    AttnSetting, ComfyUiBackend, ComfyUiBackendController, ComfyUiConfig, ComfyUiConfigArgs,
    VRamSetting,
};
pub use llamacpp::{
    ContextSize, LlamaCppBackend, LlamaCppBackendController, LlamaCppConfigArgs, LlamaCppRunConfig,
    LoadMode, OnOffAutoValue,
};

pub type LlamaCppProcessState = managed_process::ProcessState<LlamaCppRunConfig>;
