mod comfyui;
mod llamacpp;

pub use comfyui::{
    AttnSetting, ComfyUiBackend, ComfyUiBackendController, ComfyUiConfig, ComfyUiConfigArgs,
    VRamSetting,
};
pub use llamacpp::{
    ContextSize, LlamaCppBackend, LlamaCppBackendController, LlamaCppConfig, LlamaCppConfigArgs,
    OnOffValue,
};

pub type LlamaCppProcessState = managed_process::ProcessState<LlamaCppConfig>;
