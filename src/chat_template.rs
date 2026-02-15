//! Chat template support for LLM examples
//!
//! This module provides Jinja-based chat template rendering compatible with
//! HuggingFace's `tokenizer.apply_chat_template()` functionality.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use candle_examples::chat_template::{ChatTemplate, ChatTemplateOptions, Message, Conversation};
//!
//! // Load template from a model's tokenizer_config.json
//! let template = ChatTemplate::from_tokenizer_config("path/to/tokenizer_config.json")?;
//!
//! // Or use a preset for known models
//! let template = ChatTemplate::chatml(); // SmolLM, Qwen, etc.
//!
//! // Single-turn
//! let messages = vec![
//!     Message::system("You are helpful."),
//!     Message::user("Hello!"),
//! ];
//! let prompt = template.apply_for_generation(&messages)?;
//!
//! // Multi-turn conversation
//! let mut conv = Conversation::new(template, "You are helpful.");
//! let prompt = conv.user_turn("Hello!")?;
//! // ... generate response ...
//! conv.assistant_response("Hi there!");
//! let prompt = conv.user_turn("How are you?")?;
//! # Ok(())
//! # }
//! ```

use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};

/// A chat message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    // pub fn assistant(content: impl Into<String>) -> Self {
    //     Self::new("assistant", content)
    // }
}

/// Options for applying a chat template
#[derive(Debug, Clone, Default)]
pub struct ChatTemplateOptions {
    /// Add tokens that prompt the model to generate an assistant response
    pub add_generation_prompt: bool,
    /// Continue the final message instead of starting a new one (for prefilling)
    pub continue_final_message: bool,
    /// Enable thinking/reasoning mode (adds <think> tags)
    pub enable_thinking: bool,
    // Custom variables to pass to the template
    // pub extra_context: std::collections::HashMap<String, String>,
}

impl ChatTemplateOptions {
    pub fn for_generation() -> Self {
        Self {
            add_generation_prompt: true,
            ..Default::default()
        }
    }

    // pub fn for_training() -> Self {
    //     Self {
    //         add_generation_prompt: false,
    //         ..Default::default()
    //     }
    // }

    // pub fn with_thinking(mut self) -> Self {
    //     self.enable_thinking = true;
    //     self
    // }
}

/// Token configuration loaded from tokenizer_config.json
// #[derive(Debug, Clone, Default, Deserialize)]
// pub struct TokenConfig {
//     #[serde(default)]
//     pub bos_token: Option<StringOrToken>,
//     #[serde(default)]
//     pub eos_token: Option<StringOrToken>,
//     #[serde(default)]
//     pub unk_token: Option<StringOrToken>,
//     #[serde(default)]
//     pub pad_token: Option<StringOrToken>,
//     #[serde(default)]
//     pub chat_template: Option<ChatTemplateConfig>,
// }

// /// Handle both string and object token formats in tokenizer_config.json
// #[derive(Debug, Clone, Deserialize)]
// #[serde(untagged)]
// pub enum StringOrToken {
//     String(String),
//     Token { content: String },
// }

// impl StringOrToken {
//     pub fn as_str(&self) -> &str {
//         match self {
//             StringOrToken::String(s) => s,
//             StringOrToken::Token { content } => content,
//         }
//     }
// }

// impl Default for StringOrToken {
//     fn default() -> Self {
//         StringOrToken::String(String::new())
//     }
// }

/// Chat template can be a single string or multiple named templates
// #[derive(Debug, Clone, Deserialize)]
// #[serde(untagged)]
// pub enum ChatTemplateConfig {
//     Single(String),
//     Multiple(Vec<NamedTemplate>),
// }

// #[derive(Debug, Clone, Deserialize)]
// pub struct NamedTemplate {
//     pub name: String,
//     pub template: String,
// }

/// Chat template renderer using MiniJinja
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Create from a Jinja template string
    pub fn new(
        template: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        // Add the raise_exception function that HF templates use
        env.add_function("raise_exception", |msg: String| -> Result<String, _> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        env.add_template_owned("chat".to_string(), template.into())
            .map_err(|e| ChatTemplateError::TemplateError(e.to_string()))?;

        Ok(Self {
            env,
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        })
    }

    /// ChatML template used by SmolLM, Qwen, and many other models
    pub fn chatml() -> Self {
        let template = r#"
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;
        Self::new(template, "", "<|im_end|>").unwrap()
    }

    /// ChatML template with thinking/reasoning support
    //     pub fn chatml_with_thinking() -> Self {
    //         let template = r#"
    // {%- for message in messages %}
    // {{- '<|im_start|>' + message.role + '\n' + message.content | trim + '<|im_end|>\n' }}
    // {%- endfor %}
    // {%- if add_generation_prompt %}
    // {%- if enable_thinking %}
    // {{- '<|im_start|>assistant\n<think>\n' }}
    // {%- else %}
    // {{- '<|im_start|>assistant\n' }}
    // {%- endif %}
    // {%- endif %}
    // "#;
    //         Self::new(template, "", "<|im_end|>").unwrap()
    //     }
    /// Apply the chat template to messages
    pub fn apply(
        &self,
        messages: &[Message],
        options: &ChatTemplateOptions,
    ) -> Result<String, ChatTemplateError> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| ChatTemplateError::TemplateError(e.to_string()))?;

        let result = template
            .render(context! {
                messages => messages,
                add_generation_prompt => options.add_generation_prompt,
                continue_final_message => options.continue_final_message,
                enable_thinking => options.enable_thinking,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
            })
            .map_err(|e| ChatTemplateError::RenderError(e.to_string()))?;

        Ok(result.trim_start().to_string())
    }

    // Convenience method: apply with add_generation_prompt=true
    // pub fn apply_for_generation(&self, messages: &[Message]) -> Result<String, ChatTemplateError> {
    //     self.apply(messages, &ChatTemplateOptions::for_generation())
    // }
}

/// Multi-turn conversation manager
// pub struct Conversation {
//     messages: Vec<Message>,
//     template: ChatTemplate,
//     options: ChatTemplateOptions,
// }

// impl Conversation {
//     /// Create a new conversation with a system prompt
//     pub fn new(template: ChatTemplate, system_prompt: impl Into<String>) -> Self {
//         Self {
//             messages: vec![Message::system(system_prompt)],
//             template,
//             options: ChatTemplateOptions::for_generation(),
//         }
//     }

//     /// Create without a system prompt
//     pub fn without_system(template: ChatTemplate) -> Self {
//         Self {
//             messages: Vec::new(),
//             template,
//             options: ChatTemplateOptions::for_generation(),
//         }
//     }

//     /// Set options (e.g., enable thinking mode)
//     pub fn with_options(mut self, options: ChatTemplateOptions) -> Self {
//         self.options = options;
//         self
//     }

//     /// Add a user message and return the formatted prompt for generation
//     pub fn user_turn(&mut self, content: impl Into<String>) -> Result<String, ChatTemplateError> {
//         self.messages.push(Message::user(content));
//         self.template.apply(&self.messages, &self.options)
//     }

//     /// Record the assistant's response after generation
//     pub fn assistant_response(&mut self, content: impl Into<String>) {
//         self.messages.push(Message::assistant(content));
//     }

//     /// Add a message with a custom role
//     pub fn add_message(&mut self, message: Message) {
//         self.messages.push(message);
//     }

//     /// Get the conversation history
//     pub fn messages(&self) -> &[Message] {
//         &self.messages
//     }

//     /// Clear conversation history (keeps system prompt if present)
//     pub fn clear(&mut self) {
//         if let Some(first) = self.messages.first() {
//             if first.role == "system" {
//                 let system = self.messages.remove(0);
//                 self.messages.clear();
//                 self.messages.push(system);
//                 return;
//             }
//         }
//         self.messages.clear();
//     }

//     /// Format entire conversation for display (no generation prompt)
//     pub fn format_history(&self) -> Result<String, ChatTemplateError> {
//         self.template
//             .apply(&self.messages, &ChatTemplateOptions::for_training())
//     }
// }

/// Errors that can occur with chat templates
#[derive(Debug)]
pub enum ChatTemplateError {
    // IoError(String),
    // ParseError(String),
    TemplateError(String),
    RenderError(String),
    // NoTemplate,
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Self::IoError(e) => write!(f, "IO error: {}", e),
            // Self::ParseError(e) => write!(f, "Parse error: {}", e),
            Self::TemplateError(e) => write!(f, "Template error: {}", e),
            Self::RenderError(e) => write!(f, "Render error: {}", e),
            // Self::NoTemplate => write!(f, "No chat_template found in config"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}
