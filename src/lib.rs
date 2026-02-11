use crate::generation::TextGeneration;
use anyhow::{Error as E, Result};
use candle_nn::VarBuilder;
use candle_transformers::models::mimi::candle::Device;
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use tokenizers::Tokenizer;

mod generation;
mod tokenizer;
mod utils;

pub fn setup() -> Result<TextGeneration> {
    let device = Device::cuda_if_available(0)?;

    let tokenizer_filename = std::path::PathBuf::from("model/tokenizer.json");

    let filenames = utils::hub_load_local_safetensors("model", "model.safetensors.index.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let dtype = device.bf16_default_to_f32();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let config_filename = std::path::PathBuf::from("model/config.json");

    let config = std::fs::read_to_string(config_filename)?;
    let config: Phi3Config = serde_json::from_str(&config)?;

    let model = Phi3::new(&config, vb)?;

    Ok(TextGeneration::new(
        model, tokenizer, 299792458, // seed
        None,      // temperature
        None,      // top_p
        1.1,       // repeat_penalty
        64,        // repeat_last_n
        &device, false,
    ))
}
