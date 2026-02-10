use anyhow::{Error as E, Result};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
use candle_transformers::models::mimi::candle::Device;
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use crate::generation::TextGeneration;


mod utils;
mod tokenizer;
mod generation;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let start = std::time::Instant::now();
    let api = Api::new()?;

    let model_id = "microsoft/Phi-4-mini-instruct".to_string();
    let revision = "main".to_string();

    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = repo.get("tokenizer.json")?;

    let filenames = utils::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();

    let dtype = device.bf16_default_to_f32();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let config_filename = repo.get("config.json")?;
    let config = std::fs::read_to_string(config_filename)?;
    let config: Phi3Config = serde_json::from_str(&config)?;

    let model = Phi3::new(&config, vb)?;

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        299792458, // seed
        None, // temperature
        None, // top_p
        1.1,       // repeat_penalty
        64,        // repeat_last_n
        &device,
        false
    );

    pipeline.run(&"Что такое трейт объекты в расте", 5000)?;

    Ok(())
}