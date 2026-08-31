use std::ffi::OsStr;

pub enum RefImageArgs {
    PresetKrea2Edit,
    Custom(String),
}

impl AsRef<OsStr> for RefImageArgs {
    fn as_ref(&self) -> &OsStr {
        match self {
            Self::PresetKrea2Edit => "preset=krea2_edit".as_ref(),
            Self::Custom(args) => args.as_ref(),
        }
    }
}
