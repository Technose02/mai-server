use inference_backends::ContextSize;

const CTX_SIZE_HINT_T262144: &str = "max";
const CTX_SIZE_HINT_T131072: &str = "large";
const CTX_SIZE_HINT_T65536: &str = "moderate";
const CTX_SIZE_HINT_T32768: &str = "small";
const CTX_SIZE_HINT_T16384: &str = "tiny";
const CTX_SIZE_HINT_T8192: &str = "min";

pub struct ContextSizeAwareAlias(String, ContextSize);

impl ContextSizeAwareAlias {
    pub fn model(&self) -> String {
        self.0.clone()
    }
    pub fn context_size(&self) -> ContextSize {
        self.1
    }
    pub fn alias(&self) -> String {
        let suffix = match self.context_size() {
            ContextSize::T262144 => CTX_SIZE_HINT_T262144,
            ContextSize::T131072 => CTX_SIZE_HINT_T131072,
            ContextSize::T65536 => CTX_SIZE_HINT_T65536,
            ContextSize::T32768 => CTX_SIZE_HINT_T32768,
            ContextSize::T16384 => CTX_SIZE_HINT_T16384,
            ContextSize::T8192 => CTX_SIZE_HINT_T8192,
        };
        format!("{}-{suffix}", self.0)
    }
}

impl From<(String, ContextSize)> for ContextSizeAwareAlias {
    fn from(value: (String, ContextSize)) -> Self {
        Self(value.0, value.1)
    }
}

impl TryFrom<String> for ContextSizeAwareAlias {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if let Some((alias, context_size_hint)) = value.rsplit_once("-") {
            let context_size = match context_size_hint {
                CTX_SIZE_HINT_T262144 => ContextSize::T262144,
                CTX_SIZE_HINT_T131072 => ContextSize::T131072,
                CTX_SIZE_HINT_T65536 => ContextSize::T65536,
                CTX_SIZE_HINT_T32768 => ContextSize::T32768,
                CTX_SIZE_HINT_T16384 => ContextSize::T16384,
                CTX_SIZE_HINT_T8192 => ContextSize::T8192,
                _ => {
                    return Err(format!(
                        "value '{value}' contains invalid content-size-hint '{context_size_hint}'"
                    ));
                }
            };
            Ok(ContextSizeAwareAlias(alias.into(), context_size))
        } else {
            Err(format!(
                "value '{value}' does not contain context-size-hint"
            ))
        }
    }
}
