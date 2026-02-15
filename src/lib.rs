use crate::generation::TextGeneration;
use anyhow::{Error as E, Result};
use candle_transformers::models::mimi::candle::Device;
use candle_transformers::models::smol::quantized_smollm3::QuantizedModelForCausalLM;
use tokenizers::Tokenizer;

mod chat_template;
pub mod generation;
mod tokenizer;

pub struct ModelArgs {
    pub model_path: String,
    pub enable_thinking: Option<bool>,
}
impl ModelArgs {
    pub fn new(model_path: String, enable_thinking: Option<bool>) -> Self {
        Self {
            model_path,
            enable_thinking,
        }
    }
}

pub fn run() -> Result<(), E> {
    let prompt = "Rust is a great language";

    let args = ModelArgs::new("model/SmolLM3-3B-128K-Q8_0.gguf".to_string(), None);
    let mut generation_model = setup(args)?;
    generation_model.run_generation(prompt)?;

    Ok(())
}

pub fn setup(args: ModelArgs) -> Result<TextGeneration> {
    let device = Device::cuda_if_available(0)?;

    let tokenizer_filename = std::path::PathBuf::from("model/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let model_path = std::path::PathBuf::from(args.model_path);
    let model = QuantizedModelForCausalLM::from_gguf(model_path, &device)?;

    Ok(TextGeneration::new(
        &device,
        model,
        tokenizer,
        None,
        None,
        None,
        None,
        None,
        args.enable_thinking,
    ))
}
