use std::path::{Path, PathBuf};

use crate::stablediffusioncpp::{StableDiffusionError, StableDiffusionResult};

pub struct LoraDir(PathBuf);

#[cfg(target_os = "linux")]
fn softlink(original: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::unix::fs::symlink(original, link).expect("error creating symlink to lora");
}

#[cfg(target_os = "windows")]
fn softlink(original: impl AsRef<Path>, link: impl AsRef<Path>) {
    std::os::windows::fs::symlink_file(original, link).expect("error creating symlink to lora");
}

impl LoraDir {
    pub fn new(parent_dir: impl Into<PathBuf>, name: &str) -> StableDiffusionResult<LoraDir> {
        let parent_dir = parent_dir.into();
        if !parent_dir.is_dir() {
            return Err(StableDiffusionError::Custom(format!(
                "error: {} is not a valid lora-dir parent-dir",
                parent_dir.to_string_lossy()
            )));
        }
        let lora_dir = Self(parent_dir.join(name));
        if parent_dir.is_dir() {
            lora_dir.remove();
        }
        std::fs::create_dir(&lora_dir.0).expect("error creating temporary lora-dir");
        Ok(lora_dir)
    }

    pub fn add_softlink(&mut self, path: impl AsRef<Path>) -> &mut Self {
        softlink(
            &path,
            self.0.join(
                path.as_ref()
                    .file_name()
                    .unwrap_or_else(|| panic!("unable to resolve filename of lora-path")),
            ),
        );
        self
    }

    fn remove(&self) {
        if self.0.is_dir() {
            std::fs::remove_dir_all(&self.0).expect("error removing temporary lora-dir");
        }
    }
}

impl Drop for LoraDir {
    fn drop(&mut self) {
        self.remove();
    }
}

impl AsRef<Path> for LoraDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}
