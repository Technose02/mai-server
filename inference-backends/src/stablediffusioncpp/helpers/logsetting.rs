use tracing::level_filters::LevelFilter;

pub enum LogSetting {
    Nothing,
    Err(LevelFilter),
    Out(LevelFilter),
    Both {
        err_level: LevelFilter,
        out_level: LevelFilter,
    },
}
