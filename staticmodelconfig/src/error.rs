use std::{fmt::Display, path::PathBuf};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    NotAJsonFile(PathBuf),
    IoError(std::io::Error),
    DeserializationError(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAJsonFile(p) => write!(f, "the provided PathBuf {p:#?} is not a json-file"),
            Self::IoError(e) => write!(f, "nested io error: {e}"),
            Self::DeserializationError(text) => write!(f, "{text}"),
        }
    }
}

impl core::error::Error for Error {}
