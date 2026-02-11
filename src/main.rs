use anyhow::{Error, Result};

mod chat_template;
mod generation;
mod tokenizer;
mod utils;

fn main() -> Result<(), Error> {
    let prompt = "Rust is a great language";
    let mut generation_model = llm_rs::setup()?;
    generation_model.run_generation(prompt)?;

    Ok(())
}
