use anyhow::{Error as E, Result};

use candle_transformers::models::phi3::Model as Phi3;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mimi::candle::{Tensor, DType, Device, IndexOp};
use tokenizers::Tokenizer;
use std::io::Write;
use crate::tokenizer::TokenOutputStream;

const PROMPT_START: &str = "<|system|>
You are a highly knowledgeable and concise software engineering assistant. Your task is to help a human answer technical interview questions in real time. 
Guidelines:
1. Provide clear, precise, and technically correct answers. Focus on practical solutions and common best practices.
2. Keep answers short enough to be easily spoken aloud (1-3 sentences), unless a detailed explanation is explicitly requested.
3. If a question requires code, generate small, readable code snippets in the requested programming language.
4. Avoid unnecessary filler or long explanations unless asked.
5. If a question is ambiguous, ask a clarifying question briefly before giving an answer.
6. Always prioritize correctness, simplicity, and clarity.
7. Speak as if coaching someone for an interview — confident, neutral, and professional.
8. Answer only what is asked; do not provide unrelated information.
<|end|><|user|>";
const PROMPT_END: &str = "<|end|><|assistant|>";


pub struct TextGeneration {
    pub model: Phi3,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Phi3,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        verbose_prompt: bool,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            verbose_prompt,
        }
    }
    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        let current_prompt = format!("{}{}{}", PROMPT_START, prompt, PROMPT_END);
        let prompt = current_prompt.as_str();

        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;

        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the Phi3 model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }

        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };

        print!("{prompt}");
        std::io::stdout().flush()?;

        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos)?.i((.., 0, ..))?;

            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}