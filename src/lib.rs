use crate::generation::TextGeneration;
use anyhow::{Error as E, Result};
use candle_transformers::models::mimi::candle::Device;
use candle_transformers::models::smol::quantized_smollm3::QuantizedModelForCausalLM;
use tokenizers::Tokenizer;

mod chat_template;
mod generation;
mod tokenizer;
mod utils;

pub fn setup() -> Result<TextGeneration> {
    let device = Device::cuda_if_available(0)?;

    let tokenizer_filename = std::path::PathBuf::from("model/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let model_path = std::path::PathBuf::from("model/SmolLM3-3B-128K-Q8_0.gguf");
    let model = QuantizedModelForCausalLM::from_gguf(model_path, &device)?;

    Ok(TextGeneration::new(
        &device, model, tokenizer, None, None, None, None, None, None,
    ))
}
