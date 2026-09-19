#![allow(unused)]

use std::fmt::Display;

use crate::stablediffusioncpp::stablediffusionjob::backendrouting::BackendMode::Hip;

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub enum BackendMode {
    Hip,
    Cuda,
    Cpu,
    #[default]
    Auto,
}

impl Display for BackendMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendMode::Auto => write!(f, "auto"),
            BackendMode::Cpu => write!(f, "cpu"),
            BackendMode::Cuda => write!(f, "cuda"),
            BackendMode::Hip => write!(f, "hip"),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct BackendRouting {
    te: BackendMode,
    diff: BackendMode,
    vae: BackendMode,
}

impl BackendRouting {
    pub fn te(mut self, mode: BackendMode) -> Self {
        self.te = mode;
        self
    }

    pub fn diff(mut self, mode: BackendMode) -> Self {
        self.diff = mode;
        self
    }

    pub fn vae(mut self, mode: BackendMode) -> Self {
        self.vae = mode;
        self
    }

    pub fn to_arg(&self) -> Option<String> {
        let mut arg = Vec::with_capacity(3);
        if self.te != BackendMode::Auto {
            arg.push(format!("te={}", self.te));
        }
        if self.diff != BackendMode::Auto {
            arg.push(format!("diff={}", self.diff));
        }
        if self.vae != BackendMode::Auto {
            arg.push(format!("vae={}", self.vae));
        }
        if arg.is_empty() {
            None
        } else {
            Some(arg.join(","))
        }
    }
}
