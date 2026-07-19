use std::ffi::OsStr;

#[derive(Clone, Copy)]
pub enum Scheduler {
    Discrete,
    Karras,
    Exponential,
    Ays,
    Gits,
    Smoothstep,
    SgmUniform,
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

impl AsRef<OsStr> for Scheduler {
    fn as_ref(&self) -> &OsStr {
        match self {
            Scheduler::Ays => "ays".as_ref(),
            Scheduler::Beta => "beta".as_ref(),
            Scheduler::BongTangent => "bong_tangent".as_ref(),
            Scheduler::Discrete => "discrete".as_ref(),
            Scheduler::Exponential => "exponential".as_ref(),
            Scheduler::Flux => "flux".as_ref(),
            Scheduler::Flux2 => "flux2".as_ref(),
            Scheduler::Gits => "gits".as_ref(),
            Scheduler::Karras => "karras".as_ref(),
            Scheduler::KlOptimal => "kl_optimal".as_ref(),
            Scheduler::Lcm => "lcm".as_ref(),
            Scheduler::LogitNormal => "logit_normal".as_ref(),
            Scheduler::Ltx2 => "ltx2".as_ref(),
            Scheduler::SgmUniform => "sgm_uniform".as_ref(),
            Scheduler::Simple => "simple".as_ref(),
            Scheduler::Smoothstep => "smoothstep".as_ref(),
        }
    }
}
