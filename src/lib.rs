use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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
    pub system_prompt: Option<String>,
    pub enable_thinking: Option<bool>,
    pub interrupt_signal: Arc<AtomicBool>,
}
impl ModelArgs {
    pub fn new(
        model_path: String,
        system_prompt: Option<String>,
        enable_thinking: Option<bool>,
        interrupt_signal: Arc<AtomicBool>,
    ) -> Self {
        Self {
            model_path,
            system_prompt,
            enable_thinking,
            interrupt_signal,
        }
    }
}

pub fn run() -> Result<(), E> {
    let prompt = "Rust is a great language";
    let interrupt_signal = Arc::new(AtomicBool::new(false));
    let args = ModelArgs::new("model/llm.gguf".to_string(), None, None, interrupt_signal);
    let mut generation_model = setup(args, &Device::cuda_if_available(0)?)?;
    generation_model.run_generation(prompt)?;

    Ok(())
}

pub fn setup(args: ModelArgs, device: &Device) -> Result<TextGeneration> {
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
        args.system_prompt,
        args.enable_thinking,
        args.interrupt_signal,
    ))
}
