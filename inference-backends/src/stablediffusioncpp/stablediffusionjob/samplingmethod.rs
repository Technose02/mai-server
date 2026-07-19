use std::ffi::OsStr;

#[derive(Clone, Copy)]
pub enum SamplingMethod {
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

impl AsRef<OsStr> for SamplingMethod {
    fn as_ref(&self) -> &OsStr {
        match self {
            SamplingMethod::DdimTrailing => "ddim_trailing".as_ref(),
            SamplingMethod::Dpm2 => "dpm2".as_ref(),
            SamplingMethod::DpmPP2m => "dpm++2m".as_ref(),
            SamplingMethod::DpmPP2mSde => "dpm++2m_sde".as_ref(),
            SamplingMethod::DpmPP2mSdeBt => "dpm++2m_sde_bt".as_ref(),
            SamplingMethod::DpmPP2mv2 => "dpm++2mv2".as_ref(),
            SamplingMethod::DpmPP2sA => "dpm++2s_a".as_ref(),
            SamplingMethod::ErSde => "er_sde".as_ref(),
            SamplingMethod::Euler => "euler".as_ref(),
            SamplingMethod::EulerA => "euler_a".as_ref(),
            SamplingMethod::EulerACfgPp => "euler_a_cfg_pp".as_ref(),
            SamplingMethod::EulerCfgPp => "euler_cfg_pp".as_ref(),
            SamplingMethod::Heun => "heun".as_ref(),
            SamplingMethod::Ipndm => "ipndm".as_ref(),
            SamplingMethod::IpndmV => "ipndm_v".as_ref(),
            SamplingMethod::Lcm => "lcm".as_ref(),
            SamplingMethod::Res2s => "res_2s".as_ref(),
            SamplingMethod::ResMultistep => "res_multistep".as_ref(),
            SamplingMethod::Tcd => "tcd".as_ref(),
        }
    }
}
