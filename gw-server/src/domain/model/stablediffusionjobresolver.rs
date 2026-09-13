use axum::http::StatusCode;
use inference_backends::stablediffusioncpp::StableDiffusionJob;
use serde_json::json;
use std::fmt::Display;

#[derive(Debug, PartialEq)]
enum Purpose {
    T2I,
    I2I,
    TI2I,
}

#[derive(Debug)]
enum SDApi {
    AnimaTurbo,
    BooguImageTurbo,
    BooguImageEditTurbo,
    FireRedImageEditTurbo,
    Flux2Klein9b,
    FluxDev,
    FluxSchnell,
    Krea2Turbo,
    Krea2TurboEdit,
    MageFlowTurbo,
    ZImageTurbo,
}

impl TryFrom<&str> for SDApi {
    type Error = StatusCode;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "animaturbo" => Ok(SDApi::AnimaTurbo),
            "booguimageturbo" => Ok(SDApi::BooguImageTurbo),
            "booguimageeditturbo" => Ok(SDApi::BooguImageEditTurbo),
            "fireredimageeditturbo" => Ok(SDApi::FireRedImageEditTurbo),
            "flux2klein9b" => Ok(SDApi::Flux2Klein9b),
            "fluxdev" => Ok(SDApi::FluxDev),
            "fluxschnell" => Ok(SDApi::FluxSchnell),
            "krea2turbo" => Ok(SDApi::Krea2Turbo),
            "krea2turboedit" => Ok(SDApi::Krea2TurboEdit),
            "mageflowturbo" => Ok(SDApi::MageFlowTurbo),
            "zimageturbo" => Ok(SDApi::ZImageTurbo),
            _ => Err(StatusCode::NOT_FOUND),
        }
    }
}

impl Display for SDApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SDApi::AnimaTurbo => write!(f, "animaturbo"),
            SDApi::BooguImageTurbo => write!(f, "booguimageturbo"),
            SDApi::BooguImageEditTurbo => write!(f, "booguimageeditturbo"),
            SDApi::FireRedImageEditTurbo => write!(f, "fireredimageeditturbo"),
            SDApi::Flux2Klein9b => write!(f, "flux2klein9b"),
            SDApi::FluxDev => write!(f, "fluxdev"),
            SDApi::FluxSchnell => write!(f, "fluxschnell"),
            SDApi::Krea2Turbo => write!(f, "krea2turbo"),
            SDApi::Krea2TurboEdit => write!(f, "krea2turboedit"),
            SDApi::MageFlowTurbo => write!(f, "mageflowturbo"),
            SDApi::ZImageTurbo => write!(f, "zimageturbo"),
        }
    }
}

impl SDApi {
    fn create_sd_job(&self) -> StableDiffusionJob {
        match self {
            SDApi::AnimaTurbo => StableDiffusionJob::anima_turbo_job(),
            SDApi::BooguImageTurbo => StableDiffusionJob::boogu_image_turbo_job(),
            SDApi::BooguImageEditTurbo => StableDiffusionJob::boogu_image_edit_turbo_job(),
            SDApi::FireRedImageEditTurbo => StableDiffusionJob::fire_red_image_edit_8steps_job(),
            SDApi::Flux2Klein9b => StableDiffusionJob::flux2_klein_9b_job(),
            SDApi::FluxDev => StableDiffusionJob::flux_dev_job(),
            SDApi::FluxSchnell => StableDiffusionJob::flux_schnell_job(),
            SDApi::Krea2Turbo => StableDiffusionJob::krea2_turbo_job(),
            SDApi::Krea2TurboEdit => StableDiffusionJob::krea2_turbo_edit_job(),
            SDApi::MageFlowTurbo => StableDiffusionJob::mage_flow_turbo_job(),
            SDApi::ZImageTurbo => StableDiffusionJob::z_image_turbo_job(),
        }
    }

    fn list() -> Vec<Self> {
        vec![
            SDApi::AnimaTurbo,
            SDApi::BooguImageTurbo,
            SDApi::BooguImageEditTurbo,
            SDApi::FireRedImageEditTurbo,
            SDApi::Flux2Klein9b,
            SDApi::FluxDev,
            SDApi::FluxSchnell,
            SDApi::Krea2Turbo,
            SDApi::Krea2TurboEdit,
            SDApi::MageFlowTurbo,
            SDApi::ZImageTurbo,
        ]
    }

    fn purpose(&self) -> Purpose {
        match self {
            SDApi::AnimaTurbo => Purpose::T2I,
            SDApi::BooguImageTurbo => Purpose::T2I,
            SDApi::BooguImageEditTurbo => Purpose::I2I,
            SDApi::FireRedImageEditTurbo => Purpose::I2I,
            SDApi::Flux2Klein9b => Purpose::TI2I,
            SDApi::FluxDev => Purpose::T2I,
            SDApi::FluxSchnell => Purpose::T2I,
            SDApi::Krea2Turbo => Purpose::T2I,
            SDApi::Krea2TurboEdit => Purpose::I2I,
            SDApi::MageFlowTurbo => Purpose::TI2I,
            SDApi::ZImageTurbo => Purpose::T2I,
        }
    }
}

pub struct StableDiffusionJobResolver;

impl StableDiffusionJobResolver {
    pub fn create_job(name: impl AsRef<str>) -> Result<StableDiffusionJob, StatusCode> {
        Ok(SDApi::try_from(name.as_ref())?.create_sd_job())
    }

    pub fn sd_apis() -> serde_json::Value {
        json!({
            "t2i": &SDApi::list().iter().filter_map(|api| {
                if api.purpose() == Purpose::T2I || api.purpose() == Purpose::TI2I {
                    Some(format!("{api}"))
                } else {
                    None
                }
            })
            .collect::<Vec<String>>(),

            "i2i": &SDApi::list().iter().filter_map(|api| {
                if api.purpose() == Purpose::I2I || api.purpose() == Purpose::TI2I {
                    Some(format!("{api}"))
                } else {
                    None
                }
            })
            .collect::<Vec<String>>(),
        })
    }
}
