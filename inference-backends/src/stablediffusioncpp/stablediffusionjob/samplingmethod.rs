use serde::{Deserialize, Serialize, de::Visitor};

#[derive(Debug, Clone, Copy, Default)]
pub enum SamplingMethod {
    #[default]
    Euler,
    EulerA,
    Heun,
    Dpm2,
    DpmPP2sA,
    DpmPP2m,
    DpmPP2mv2,
    DpmPP2mSde,
    DpmPP2mSdeBt,
    Ipndm,
    IpndmV,
    Lcm,
    DdimTrailing,
    Tcd,
    ResMultistep,
    Res2s,
    ErSde,
    EulerCfgPp,
    EulerACfgPp,
}

impl AsRef<str> for SamplingMethod {
    fn as_ref(&self) -> &str {
        match self {
            SamplingMethod::DdimTrailing => "ddim_trailing",
            SamplingMethod::Dpm2 => "dpm2",
            SamplingMethod::DpmPP2m => "dpm++2m",
            SamplingMethod::DpmPP2mSde => "dpm++2m_sde",
            SamplingMethod::DpmPP2mSdeBt => "dpm++2m_sde_bt",
            SamplingMethod::DpmPP2mv2 => "dpm++2mv2",
            SamplingMethod::DpmPP2sA => "dpm++2s_a",
            SamplingMethod::ErSde => "er_sde",
            SamplingMethod::Euler => "euler",
            SamplingMethod::EulerA => "euler_a",
            SamplingMethod::EulerACfgPp => "euler_a_cfg_pp",
            SamplingMethod::EulerCfgPp => "euler_cfg_pp",
            SamplingMethod::Heun => "heun",
            SamplingMethod::Ipndm => "ipndm",
            SamplingMethod::IpndmV => "ipndm_v",
            SamplingMethod::Lcm => "lcm",
            SamplingMethod::Res2s => "res_2s",
            SamplingMethod::ResMultistep => "res_multistep",
            SamplingMethod::Tcd => "tcd",
        }
    }
}

impl TryFrom<&str> for SamplingMethod {
    type Error = Box<dyn core::error::Error>;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "ddim_trailing" => Ok(SamplingMethod::DdimTrailing),
            "dpm2" => Ok(SamplingMethod::Dpm2),
            "dpm++2m" => Ok(SamplingMethod::DpmPP2m),
            "dpm++2m_sde" => Ok(SamplingMethod::DpmPP2mSde),
            "dpm++2m_sde_bt" => Ok(SamplingMethod::DpmPP2mSdeBt),
            "dpm++2mv2" => Ok(SamplingMethod::DpmPP2mv2),
            "dpm++2s_a" => Ok(SamplingMethod::DpmPP2sA),
            "er_sde" => Ok(SamplingMethod::ErSde),
            "euler" => Ok(SamplingMethod::Euler),
            "euler_a" => Ok(SamplingMethod::EulerA),
            "euler_a_cfg_pp" => Ok(SamplingMethod::EulerACfgPp),
            "euler_cfg_pp" => Ok(SamplingMethod::EulerCfgPp),
            "heun" => Ok(SamplingMethod::Heun),
            "ipndm" => Ok(SamplingMethod::Ipndm),
            "ipndm_v" => Ok(SamplingMethod::IpndmV),
            "lcm" => Ok(SamplingMethod::Lcm),
            "res_2s" => Ok(SamplingMethod::Res2s),
            "res_multistep" => Ok(SamplingMethod::ResMultistep),
            "tcd" => Ok(SamplingMethod::Tcd),
            _ => Err(format!("invalid value '{value}' for SamplingMethod").into()),
        }
    }
}

impl Serialize for SamplingMethod {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_ref())
    }
}

struct SamplingMethodVisitor;

impl<'de> Visitor<'de> for SamplingMethodVisitor {
    type Value = SamplingMethod;

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<SamplingMethod>::try_into(v.as_ref()).map_err(|e| E::custom(e.to_string()))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TryInto::<SamplingMethod>::try_into(v).map_err(|e| E::custom(e.to_string()))
    }

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("expected any String")
    }
}

impl<'de> Deserialize<'de> for SamplingMethod {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(SamplingMethodVisitor)
    }
}
