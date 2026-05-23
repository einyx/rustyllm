use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("not enough disk space at {path}: need {need_gb:.2} GB, free {free_gb:.2} GB")]
    NotEnoughSpace {
        path: PathBuf,
        need_gb: f64,
        free_gb: f64,
    },
    #[error("layer shard not found: {0}")]
    MissingShard(String),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;
