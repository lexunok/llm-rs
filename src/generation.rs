use crate::chat_template::{ChatTemplate, ChatTemplateOptions, Message};
use crate::tokenizer::TokenOutputStream;
use anyhow::{Error as E, Result};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::mimi::candle::{DType, Device, Tensor};
use candle_transformers::models::smol::quantized_smollm3::QuantizedModelForCausalLM;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;

pub struct TextGeneration {
    pub model: QuantizedModelForCausalLM,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub enable_thinking: bool,
    pub top_p: f64,
    pub temperature: f64,
    pub sample_len: usize,
    pub system_prompt: String,
    pub interrupt_signal: Arc<AtomicBool>,
}

impl TextGeneration {
    pub fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }
    pub fn run_generation(&mut self, prompt_str: &str) -> Result<()> {
        self.interrupt_signal.store(false, Ordering::Relaxed);
        let formatted_prompt = self.format_prompt(&prompt_str);
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(formatted_prompt.as_str(), false)
            .map_err(E::msg)?;
        let tokens = tokens.get_ids();

        let sampling = Sampling::TopP {
            p: self.top_p,
            temperature: self.temperature,
        };

        let mut logits_processor = LogitsProcessor::from_sampling(299792458, sampling);

        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let mut next_token = logits_processor.sample(&logits)?;

        let eos_token = self.get_eos_token();

        let mut all_tokens = vec![next_token];
        println!("\nОтвет:");
        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let to_sample = self.sample_len.saturating_sub(1);

        for index in 0..to_sample {
            if self.interrupt_signal.load(Ordering::Relaxed) {
                println!("\n[Генерация прервана]");
                self.clear_cache();
                break;
            }
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;

            let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
            let logits = candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &all_tokens[start_at..],
            )?;

            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            if next_token == eos_token {
                break;
            }
        }

        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        println!();
        Ok(())
    }
    fn get_eos_token(&self) -> u32 {
        let vocab = self.tokenizer.tokenizer().get_vocab(true);
        if let Some(&eos_id) = vocab.get("<|im_end|>") {
            return eos_id;
        }
        if let Some(&eos_id) = vocab.get("<|endoftext|>") {
            return eos_id;
        }

        128012 // Default SmolLM3 EOS token
    }
    fn format_prompt(&self, prompt: &str) -> String {
        // let template = if self.enable_thinking {
        //     ChatTemplate::chatml_with_thinking()
        // } else {
        //     ChatTemplate::chatml()
        // };
        let template = ChatTemplate::chatml();
        // Build system message with SmolLM3's metadata format
        let now = chrono::Local::now();
        let today_date = now.format("%d %B %Y").to_string();

        let reasoning_mode = "/no_think";
        // let reasoning_mode = if self.enable_thinking {
        //     "/think"
        // } else {
        //     "/no_think"
        // };

        let system_content = format!(
            "## Metadata\n\n\
             Knowledge Cutoff Date: February 2026\n\
             Today Date: {}\n\
             Reasoning Mode: {}\n\n\
             ## Custom Instructions\n\n\
             {}",
            today_date, reasoning_mode, self.system_prompt
        );

        let messages = vec![Message::system(system_content), Message::user(prompt)];

        let options = ChatTemplateOptions::for_generation();
        // let options = if self.enable_thinking {
        //     ChatTemplateOptions::for_generation().with_thinking()
        // } else {
        //     ChatTemplateOptions::for_generation()
        // };

        template.apply(&messages, &options).unwrap()
    }
    pub fn new(
        device: &Device,
        model: QuantizedModelForCausalLM,
        tokenizer: Tokenizer,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        sample_len: Option<usize>,
        system_prompt: Option<String>,
        enable_thinking: Option<bool>,
        interrupt_signal: Arc<AtomicBool>,
    ) -> Self {
        Self {
            device: device.clone(),
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            repeat_penalty: repeat_penalty.unwrap_or(1.1),
            repeat_last_n: repeat_last_n.unwrap_or(64),
            sample_len: sample_len.unwrap_or(1000),
            temperature: temp.unwrap_or(0.6),
            top_p: top_p.unwrap_or(0.5),
            system_prompt: system_prompt.unwrap_or("You are a helpful assistant".to_string()),
            enable_thinking: enable_thinking.unwrap_or(false),
            interrupt_signal: interrupt_signal,
        }
    }
}
