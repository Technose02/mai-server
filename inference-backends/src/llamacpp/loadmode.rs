use std::fmt::Display;

use serde::{Deserialize, Serialize, de::Visitor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadMode {
    Auto,
    Mmap,
    Mlock,
    MmapPlusMlock,
    None,
    Dio,
}

impl AsRef<str> for LoadMode {
    fn as_ref(&self) -> &str {
        match self {
            LoadMode::Auto => "auto",
            LoadMode::Mmap => "mmap",
            LoadMode::Mlock => "mlock",
            LoadMode::MmapPlusMlock => "mmap+mlock",
            LoadMode::None => "none",
            LoadMode::Dio => "dio",
        }
    }
}

impl Display for LoadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl Serialize for LoadMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

struct LoadModeVisitor;
impl LoadModeVisitor {
    fn values() -> &'static str {
        "<auto|mmap|mlock|mmap+mlock|dio|none>"
    }
}
impl<'de> Visitor<'de> for LoadModeVisitor {
    type Value = LoadMode;

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match v {
            "auto" => Ok(LoadMode::Auto),
            "mmap" => Ok(LoadMode::Mmap),
            "mlock" => Ok(LoadMode::Mlock),
            "mmap+mlock" => Ok(LoadMode::MmapPlusMlock),
            "none" => Ok(LoadMode::None),
            "dio" => Ok(LoadMode::Dio),
            _ => Err(E::custom(format!(
                "invalid value \"{v}\"; expected \"{}\"",
                LoadModeVisitor::values()
            ))),
        }
    }

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_fmt(format_args!(
            "expected any of \"{}\"",
            LoadModeVisitor::values()
        ))
    }
}

impl<'de> Deserialize<'de> for LoadMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(LoadModeVisitor)
    }
}
