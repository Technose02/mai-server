#[derive(Clone, Copy, Debug, Default)]
pub enum FlashAttentionMode {
    Full,
    DiffusionOnly,
    #[default]
    None,
}
