use serde::{Deserialize, Serialize, de::Visitor};

#[derive(Debug, Clone)]
pub enum RefImageArgs {
    PresetKrea2Edit,
    Custom(String),
}

impl AsRef<str> for RefImageArgs {
    fn as_ref(&self) -> &str {
        match self {
            Self::PresetKrea2Edit => "preset=krea2_edit",
            Self::Custom(args) => args,
        }
    }
}

impl TryFrom<&str> for RefImageArgs {
    type Error = Box<dyn core::error::Error>;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "preset=krea2_edit" => Ok(Self::PresetKrea2Edit),
            custom_args => Ok(Self::Custom(custom_args.into())),
        }
    }
}

impl Serialize for RefImageArgs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

struct RefImageArgsVisitor;

impl<'de> Visitor<'de> for RefImageArgsVisitor {
    type Value = RefImageArgs;

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<RefImageArgs>::try_into(v.as_ref()).map_err(|e| E::custom(e.to_string()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<RefImageArgs>::try_into(v).map_err(|e| E::custom(e.to_string()))
    }

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("expected any String")
    }
}

impl<'de> Deserialize<'de> for RefImageArgs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(RefImageArgsVisitor)
    }
}
