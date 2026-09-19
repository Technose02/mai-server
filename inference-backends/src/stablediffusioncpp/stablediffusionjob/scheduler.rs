use serde::{Deserialize, Serialize, de::Visitor};

#[derive(Debug, Clone, Copy, Default)]
pub enum Scheduler {
    Discrete,
    Karras,
    Exponential,
    Ays,
    Gits,
    Smoothstep,
    SgmUniform,
    #[default]
    Simple,
    KlOptimal,
    Lcm,
    BongTangent,
    Ltx2,
    LogitNormal,
    Flux2,
    Flux,
    Beta, /*--scheduler                              [, , , , ,
          , , , , , , ,
          , , , ], alias: normal=discrete, default:
          model-specific */
}

impl TryFrom<&str> for Scheduler {
    type Error = Box<dyn core::error::Error>;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "ays" => Ok(Scheduler::Ays),
            "beta" => Ok(Scheduler::Beta),
            "bong_tangent" => Ok(Scheduler::BongTangent),
            "discrete" => Ok(Scheduler::Discrete),
            "exponential" => Ok(Scheduler::Exponential),
            "flux" => Ok(Scheduler::Flux),
            "flux2" => Ok(Scheduler::Flux2),
            "gits" => Ok(Scheduler::Gits),
            "karras" => Ok(Scheduler::Karras),
            "kl_optimal" => Ok(Scheduler::KlOptimal),
            "lcm" => Ok(Scheduler::Lcm),
            "logit_normal" => Ok(Scheduler::LogitNormal),
            "ltx2" => Ok(Scheduler::Ltx2),
            "sgm_uniform" => Ok(Scheduler::SgmUniform),
            "simple" => Ok(Scheduler::Simple),
            "smoothstep" => Ok(Scheduler::Smoothstep),
            _ => Err(format!("invalid value '{value}' for Scheduler").into()),
        }
    }
}

impl AsRef<str> for Scheduler {
    fn as_ref(&self) -> &str {
        match self {
            Scheduler::Ays => "ays",
            Scheduler::Beta => "beta",
            Scheduler::BongTangent => "bong_tangent",
            Scheduler::Discrete => "discrete",
            Scheduler::Exponential => "exponential",
            Scheduler::Flux => "flux",
            Scheduler::Flux2 => "flux2",
            Scheduler::Gits => "gits",
            Scheduler::Karras => "karras",
            Scheduler::KlOptimal => "kl_optimal",
            Scheduler::Lcm => "lcm",
            Scheduler::LogitNormal => "logit_normal",
            Scheduler::Ltx2 => "ltx2",
            Scheduler::SgmUniform => "sgm_uniform",
            Scheduler::Simple => "simple",
            Scheduler::Smoothstep => "smoothstep",
        }
    }
}

impl Serialize for Scheduler {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

struct SchedulerVisitor;

impl<'de> Visitor<'de> for SchedulerVisitor {
    type Value = Scheduler;

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<Scheduler>::try_into(v.as_ref()).map_err(|e| E::custom(e.to_string()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<Scheduler>::try_into(v).map_err(|e| E::custom(e.to_string()))
    }

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("expected any String")
    }
}

impl<'de> Deserialize<'de> for Scheduler {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(SchedulerVisitor)
    }
}
