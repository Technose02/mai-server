use std::{collections::HashMap, sync::Arc};
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq)]

pub struct ComfyUiConfig {
    pub env_handle: Arc<HashMap<String, String>>,
    pub args_handle: Arc<ComfyUiConfigArgs>,
}

impl ComfyUiConfig {
    pub fn apply_args(&self, cmd: &mut Command) {
        self.args_handle.apply(cmd);
    }

    pub fn apply_env(&self, cmd: &mut Command) {
        for (key, val) in self.env_handle.iter() {
            cmd.env(key, val);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComfyUiConfigArgs {
    pub fp32_vae: bool,
    pub use_flash_attention: bool,
    pub vram_setting: Option<VRamSetting>,
    pub attn_setting: Option<AttnSetting>,
}

impl ComfyUiConfigArgs {
    pub fn apply(&self, cmd: &mut Command) {
        if self.fp32_vae {
            cmd.arg("--fp32-vae");
        }
        if self.use_flash_attention {
            cmd.arg("--use-flash-attention");
        }

        match self.vram_setting {
            Some(VRamSetting::GpuOnly) => {
                cmd.arg("--gpu-only");
            }
            Some(VRamSetting::HighVram) => {
                cmd.arg("--highvram");
            }
            Some(VRamSetting::NormalVram) => {
                cmd.arg("--normalvram");
            }
            Some(VRamSetting::LowVram) => {
                cmd.arg("--lowvram");
            }
            Some(VRamSetting::NoVram) => {
                cmd.arg("--novram");
            }
            Some(VRamSetting::Cpu) => {
                cmd.arg("--cpu");
            }
            None => {}
        }

        match self.attn_setting {
            Some(AttnSetting::SplitCrossAttention) => {
                cmd.arg("--use-split-cross-attention");
            }
            Some(AttnSetting::QuadCrossAttention) => {
                cmd.arg("--use-quad-cross-attention");
            }
            Some(AttnSetting::PytorchCrossAttention) => {
                cmd.arg("--use-pytorch-cross-attention");
            }
            Some(AttnSetting::SageAttention) => {
                cmd.arg("--use-sage-attention");
            }
            Some(AttnSetting::FlashAttention) => {
                cmd.arg("--use-flash-attention");
            }
            None => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VRamSetting {
    GpuOnly,
    HighVram,
    NormalVram,
    LowVram,
    NoVram,
    Cpu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttnSetting {
    SplitCrossAttention,
    QuadCrossAttention,
    PytorchCrossAttention,
    SageAttention,
    FlashAttention,
}
