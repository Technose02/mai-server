mod comfyui;
mod llamacpp;

pub use comfyui::{
    AttnSetting, ComfyUiBackend, ComfyUiBackendController, ComfyUiConfig, ComfyUiConfigArgs,
    VRamSetting,
};
pub use llamacpp::{
    LlamaCppBackend, LlamaCppBackendController, LlamaCppConfig, LlamaCppConfigArgs,
};

pub type LlamaCppProcessState = managed_process::ProcessState<LlamaCppConfig>;
