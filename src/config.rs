use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read configuration file at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Failed to parse TOML configuration from {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("Invalid environment variable override for {key}: {val}")]
    EnvOverride { key: String, val: String },
    #[error("Invalid configuration: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContinuationMode {
    Prefix,
    /// Default because the default `base_url` is Google's OpenAI-compatibility
    /// endpoint, which rejects any request whose messages end with an assistant
    /// turn ("Requests ending with a model turn are not supported") — exactly
    /// what `Prefix` builds. A default that cannot resume against the default
    /// upstream is not a usable default. See §2 of the spec.
    #[default]
    Instructed,
}

impl std::str::FromStr for ContinuationMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "prefix" => Ok(Self::Prefix),
            "instructed" => Ok(Self::Instructed),
            other => Err(format!(
                "Unknown continuation mode: '{other}'. Expected 'prefix' or 'instructed'"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Maximum accepted downstream request body.
    ///
    /// axum defaults to 2 MiB, which a long conversation or a pasted document
    /// exceeds easily — and a rejected prompt is a far more likely failure here
    /// than resource exhaustion, since the proxy binds to loopback by default.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Browser origins permitted to call this proxy. Empty (the default) sends
    /// no CORS headers at all.
    ///
    /// Jan.ai and other desktop clients issue native HTTP requests with no
    /// `Origin`, so they need nothing here. Permissive CORS would only widen the
    /// surface: it would let any web page you happen to visit spend the
    /// configured account's quota through your loopback listener.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    /// Chat completions handled at once. Beyond this, requests are refused with
    /// 503 rather than queued: each in-flight request holds an upstream
    /// connection and up to `max_accumulated_bytes` of state, so unbounded
    /// admission turns a burst into memory exhaustion.
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
}

fn default_bind() -> String {
    "127.0.0.1:8787".to_string()
}

fn default_max_body_bytes() -> usize {
    32 * 1024 * 1024
}
fn default_max_concurrent_requests() -> usize {
    64
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            max_body_bytes: default_max_body_bytes(),
            allowed_origins: Vec::new(),
            max_concurrent_requests: default_max_concurrent_requests(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamConfig {
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    /// Cap on a non-streaming or `/v1/models` response body, and on an error
    /// body read before classification. Timeouts bound how long an upstream may
    /// take, not how much it may send.
    #[serde(default = "default_max_response_bytes")]
    pub max_response_bytes: usize,
}

fn default_base_url() -> String {
    "https://generativelanguage.googleapis.com/v1beta/openai".to_string()
}

fn default_api_key_env() -> String {
    "GEMINI_API_KEY".to_string()
}

fn default_connect_timeout_secs() -> u64 {
    10
}
fn default_max_response_bytes() -> usize {
    32 * 1024 * 1024
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            api_key_env: default_api_key_env(),
            api_key: None,
            connect_timeout_secs: default_connect_timeout_secs(),
            max_response_bytes: default_max_response_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResumeConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub mode: ContinuationMode,
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_max_wall_clock_secs")]
    pub max_wall_clock_secs: u64,
    /// Deadline used instead of `max_wall_clock_secs` while every failure so far
    /// has been the upstream saying it is busy (§7.3). Only ever extends: the
    /// effective deadline is the larger of the two, so setting this below
    /// `max_wall_clock_secs` has no effect.
    #[serde(default = "default_overloaded_max_wall_clock_secs")]
    pub overloaded_max_wall_clock_secs: u64,
    #[serde(default = "default_backoff_initial_ms")]
    pub backoff_initial_ms: u64,
    #[serde(default = "default_backoff_max_ms")]
    pub backoff_max_ms: u64,
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    #[serde(default = "default_backoff_jitter")]
    pub backoff_jitter: f64,
    #[serde(default = "default_dedup_window_chars")]
    pub dedup_window_chars: usize,
    #[serde(default = "default_first_chunk_timeout_secs")]
    pub first_chunk_timeout_secs: u64,
    #[serde(default = "default_idle_timeout_secs")]
    pub idle_timeout_secs: u64,
    #[serde(default)]
    pub resume_on_length_finish: bool,
    /// Give up after this many consecutive attempts that fail the same way while
    /// emitting nothing new (§7.5). Congestion is exempt. 0 disables the check.
    #[serde(default = "default_max_no_progress_attempts")]
    pub max_no_progress_attempts: u32,
    /// Cap on the text accumulated across all attempts of one response.
    ///
    /// Deliberately far above any real generation — 32 MiB is on the order of
    /// eight million tokens — so it protects memory against a runaway or
    /// malicious upstream without ever truncating legitimate long output.
    #[serde(default = "default_max_accumulated_bytes")]
    pub max_accumulated_bytes: usize,
}

fn default_true() -> bool {
    true
}
fn default_max_attempts() -> u32 {
    6
}
fn default_max_wall_clock_secs() -> u64 {
    900
}
fn default_overloaded_max_wall_clock_secs() -> u64 {
    300
}
fn default_backoff_initial_ms() -> u64 {
    500
}
fn default_backoff_max_ms() -> u64 {
    30000
}
fn default_backoff_multiplier() -> f64 {
    2.0
}
fn default_backoff_jitter() -> f64 {
    0.25
}
fn default_dedup_window_chars() -> usize {
    240
}
fn default_first_chunk_timeout_secs() -> u64 {
    120
}
fn default_idle_timeout_secs() -> u64 {
    45
}
fn default_max_no_progress_attempts() -> u32 {
    3
}
fn default_max_accumulated_bytes() -> usize {
    32 * 1024 * 1024
}

impl Default for ResumeConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            mode: ContinuationMode::default(),
            max_attempts: default_max_attempts(),
            max_wall_clock_secs: default_max_wall_clock_secs(),
            overloaded_max_wall_clock_secs: default_overloaded_max_wall_clock_secs(),
            backoff_initial_ms: default_backoff_initial_ms(),
            backoff_max_ms: default_backoff_max_ms(),
            backoff_multiplier: default_backoff_multiplier(),
            backoff_jitter: default_backoff_jitter(),
            dedup_window_chars: default_dedup_window_chars(),
            first_chunk_timeout_secs: default_first_chunk_timeout_secs(),
            idle_timeout_secs: default_idle_timeout_secs(),
            resume_on_length_finish: false,
            max_no_progress_attempts: default_max_no_progress_attempts(),
            max_accumulated_bytes: default_max_accumulated_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    #[serde(default = "default_true")]
    pub buffer_tool_calls: bool,
    #[serde(default)]
    pub disable_resume_for_tool_responses: bool,
    /// Cap on the assembled arguments of a single buffered tool call.
    #[serde(default = "default_max_tool_call_bytes")]
    pub max_tool_call_bytes: usize,
}

fn default_max_tool_call_bytes() -> usize {
    4 * 1024 * 1024
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            buffer_tool_calls: true,
            disable_resume_for_tool_responses: false,
            max_tool_call_bytes: default_max_tool_call_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub log_bodies: bool,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            log_bodies: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub resume: ResumeConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub log: LogConfig,
}

impl Config {
    pub fn default_config_path() -> PathBuf {
        if let Ok(path) = env::var("LLM_PROXY_CONFIG") {
            return expand_tilde(&path);
        }
        expand_tilde("~/.config/llm-proxy/llm-proxy.toml")
    }

    pub fn load(explicit_path: Option<&Path>) -> Result<Self, ConfigError> {
        // A path named through `$LLM_PROXY_CONFIG` was chosen deliberately, so a
        // typo in it must fail loudly rather than silently routing traffic to
        // the built-in default upstream. Only the implicit default path may fall
        // back to `Config::default()`.
        let env_path = env::var("LLM_PROXY_CONFIG").ok().map(|p| expand_tilde(&p));
        let (config_path, is_explicit) = match (explicit_path, env_path) {
            (Some(p), _) => (expand_path(p), true),
            (None, Some(p)) => (p, true),
            (None, None) => (expand_tilde("~/.config/llm-proxy/llm-proxy.toml"), false),
        };

        let mut config = if config_path.exists() {
            let content =
                std::fs::read_to_string(&config_path).map_err(|source| ConfigError::Io {
                    path: config_path.clone(),
                    source,
                })?;
            toml::from_str::<Config>(&content).map_err(|source| ConfigError::Parse {
                path: config_path.clone(),
                source,
            })?
        } else if is_explicit {
            return Err(ConfigError::Io {
                path: config_path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Config file not found at {}", config_path.display()),
                ),
            });
        } else {
            Config::default()
        };

        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Resolve the upstream key, trimmed.
    ///
    /// Surrounding whitespace is stripped rather than merely tested for: a key
    /// read from a file or a here-doc usually carries a trailing newline, which
    /// is not a legal HTTP header value and fails at request time.
    pub fn get_upstream_api_key(&self) -> Option<String> {
        if let Some(key) = self.upstream.api_key.as_deref() {
            let key = key.trim();
            if !key.is_empty() {
                return Some(key.to_string());
            }
        }
        if !self.upstream.api_key_env.is_empty() {
            if let Ok(val) = env::var(&self.upstream.api_key_env) {
                let val = val.trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
        }
        None
    }

    /// Reject values that parse but cannot work.
    ///
    /// Every field here is individually well-typed, so serde and the env
    /// overrides both accept nonsense: `max_attempts = 0` produces an empty
    /// successful stream without ever contacting the upstream, a non-finite
    /// multiplier propagates NaN into the backoff, and a malformed `base_url`
    /// fails identically on every attempt until the retry budget is gone.
    /// Failing at startup turns all of those into one clear message.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |msg: String| Err(ConfigError::Validation(msg));

        if self.resume.max_attempts == 0 {
            return invalid("resume.max_attempts must be at least 1".into());
        }
        if self.server.max_concurrent_requests == 0 {
            return invalid("server.max_concurrent_requests must be at least 1".into());
        }

        for (name, value) in [
            ("server.max_body_bytes", self.server.max_body_bytes),
            (
                "upstream.max_response_bytes",
                self.upstream.max_response_bytes,
            ),
            ("resume.dedup_window_chars", self.resume.dedup_window_chars),
            (
                "resume.max_accumulated_bytes",
                self.resume.max_accumulated_bytes,
            ),
            ("tools.max_tool_call_bytes", self.tools.max_tool_call_bytes),
        ] {
            if value == 0 {
                return invalid(format!("{name} must be greater than 0"));
            }
        }

        // A day is far past any sane setting and keeps deadline arithmetic well
        // clear of `Instant` overflow.
        const MAX_SECS: u64 = 86_400;
        for (name, value) in [
            (
                "upstream.connect_timeout_secs",
                self.upstream.connect_timeout_secs,
            ),
            (
                "resume.max_wall_clock_secs",
                self.resume.max_wall_clock_secs,
            ),
            (
                "resume.overloaded_max_wall_clock_secs",
                self.resume.overloaded_max_wall_clock_secs,
            ),
            (
                "resume.first_chunk_timeout_secs",
                self.resume.first_chunk_timeout_secs,
            ),
            ("resume.idle_timeout_secs", self.resume.idle_timeout_secs),
        ] {
            if value == 0 {
                return invalid(format!("{name} must be greater than 0"));
            }
            if value > MAX_SECS {
                return invalid(format!("{name} must be at most {MAX_SECS} seconds"));
            }
        }

        if self.resume.backoff_initial_ms == 0 || self.resume.backoff_max_ms == 0 {
            return invalid("resume backoff durations must be greater than 0".into());
        }
        if self.resume.backoff_max_ms < self.resume.backoff_initial_ms {
            return invalid(
                "resume.backoff_max_ms must be at least resume.backoff_initial_ms".into(),
            );
        }
        if !self.resume.backoff_multiplier.is_finite() || self.resume.backoff_multiplier < 1.0 {
            return invalid("resume.backoff_multiplier must be finite and at least 1.0".into());
        }
        if !self.resume.backoff_jitter.is_finite()
            || !(0.0..=1.0).contains(&self.resume.backoff_jitter)
        {
            return invalid("resume.backoff_jitter must be between 0.0 and 1.0".into());
        }

        if self.upstream.base_url.trim().is_empty() {
            return invalid("upstream.base_url must not be empty".into());
        }
        let url = reqwest::Url::parse(&self.upstream.base_url)
            .map_err(|e| ConfigError::Validation(format!("upstream.base_url is not a URL: {e}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return invalid(format!(
                "upstream.base_url scheme must be http or https, got '{}'",
                url.scheme()
            ));
        }

        if let Some(key) = self.get_upstream_api_key() {
            if reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")).is_err() {
                return invalid(
                    "the upstream API key contains characters that cannot be sent in a header"
                        .into(),
                );
            }
        }

        Ok(())
    }

    fn apply_env_overrides(&mut self) -> Result<(), ConfigError> {
        // Server
        if let Ok(val) = env::var("LLM_PROXY_SERVER_BIND") {
            self.server.bind = val;
        }
        if let Ok(val) = env::var("LLM_PROXY_SERVER_MAX_BODY_BYTES") {
            self.server.max_body_bytes = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_SERVER_MAX_BODY_BYTES".to_string(),
                val,
            })?;
        }

        // Upstream
        if let Ok(val) = env::var("LLM_PROXY_UPSTREAM_BASE_URL") {
            self.upstream.base_url = val;
        }
        if let Ok(val) = env::var("LLM_PROXY_UPSTREAM_API_KEY_ENV") {
            self.upstream.api_key_env = val;
        }
        if let Ok(val) = env::var("LLM_PROXY_UPSTREAM_API_KEY") {
            self.upstream.api_key = Some(val);
        }
        if let Ok(val) = env::var("LLM_PROXY_SERVER_MAX_CONCURRENT_REQUESTS") {
            self.server.max_concurrent_requests =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_SERVER_MAX_CONCURRENT_REQUESTS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_SERVER_ALLOWED_ORIGINS") {
            self.server.allowed_origins = val
                .split(',')
                .map(|o| o.trim().to_string())
                .filter(|o| !o.is_empty())
                .collect();
        }
        if let Ok(val) = env::var("LLM_PROXY_UPSTREAM_MAX_RESPONSE_BYTES") {
            self.upstream.max_response_bytes =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_UPSTREAM_MAX_RESPONSE_BYTES".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_MAX_ACCUMULATED_BYTES") {
            self.resume.max_accumulated_bytes =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_MAX_ACCUMULATED_BYTES".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_TOOLS_MAX_TOOL_CALL_BYTES") {
            self.tools.max_tool_call_bytes = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_TOOLS_MAX_TOOL_CALL_BYTES".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_UPSTREAM_CONNECT_TIMEOUT_SECS") {
            self.upstream.connect_timeout_secs =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_UPSTREAM_CONNECT_TIMEOUT_SECS".to_string(),
                    val,
                })?;
        }

        // Resume
        if let Ok(val) = env::var("LLM_PROXY_RESUME_ENABLED") {
            self.resume.enabled = parse_bool(&val).map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_ENABLED".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_MODE") {
            self.resume.mode = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_MODE".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_MAX_ATTEMPTS") {
            self.resume.max_attempts = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_MAX_ATTEMPTS".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_MAX_WALL_CLOCK_SECS") {
            self.resume.max_wall_clock_secs =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_MAX_WALL_CLOCK_SECS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_OVERLOADED_MAX_WALL_CLOCK_SECS") {
            self.resume.overloaded_max_wall_clock_secs =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_OVERLOADED_MAX_WALL_CLOCK_SECS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_BACKOFF_INITIAL_MS") {
            self.resume.backoff_initial_ms = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_BACKOFF_INITIAL_MS".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_BACKOFF_MAX_MS") {
            self.resume.backoff_max_ms = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_BACKOFF_MAX_MS".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_BACKOFF_MULTIPLIER") {
            self.resume.backoff_multiplier = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_BACKOFF_MULTIPLIER".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_BACKOFF_JITTER") {
            self.resume.backoff_jitter = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_BACKOFF_JITTER".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_DEDUP_WINDOW_CHARS") {
            self.resume.dedup_window_chars = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_DEDUP_WINDOW_CHARS".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_FIRST_CHUNK_TIMEOUT_SECS") {
            self.resume.first_chunk_timeout_secs =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_FIRST_CHUNK_TIMEOUT_SECS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_IDLE_TIMEOUT_SECS") {
            self.resume.idle_timeout_secs = val.parse().map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_RESUME_IDLE_TIMEOUT_SECS".to_string(),
                val,
            })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_MAX_NO_PROGRESS_ATTEMPTS") {
            self.resume.max_no_progress_attempts =
                val.parse().map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_MAX_NO_PROGRESS_ATTEMPTS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_RESUME_RESUME_ON_LENGTH_FINISH") {
            self.resume.resume_on_length_finish =
                parse_bool(&val).map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_RESUME_RESUME_ON_LENGTH_FINISH".to_string(),
                    val,
                })?;
        }

        // Tools
        if let Ok(val) = env::var("LLM_PROXY_TOOLS_BUFFER_TOOL_CALLS") {
            self.tools.buffer_tool_calls =
                parse_bool(&val).map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_TOOLS_BUFFER_TOOL_CALLS".to_string(),
                    val,
                })?;
        }
        if let Ok(val) = env::var("LLM_PROXY_TOOLS_DISABLE_RESUME_FOR_TOOL_RESPONSES") {
            self.tools.disable_resume_for_tool_responses =
                parse_bool(&val).map_err(|_| ConfigError::EnvOverride {
                    key: "LLM_PROXY_TOOLS_DISABLE_RESUME_FOR_TOOL_RESPONSES".to_string(),
                    val,
                })?;
        }

        // Log
        if let Ok(val) = env::var("LLM_PROXY_LOG_LEVEL") {
            self.log.level = val;
        }
        if let Ok(val) = env::var("LLM_PROXY_LOG_LOG_BODIES") {
            self.log.log_bodies = parse_bool(&val).map_err(|_| ConfigError::EnvOverride {
                key: "LLM_PROXY_LOG_LOG_BODIES".to_string(),
                val,
            })?;
        }

        Ok(())
    }
}

fn parse_bool(val: &str) -> Result<bool, ()> {
    match val.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(()),
    }
}

/// True when `base` already ends with `path` as a whole path segment.
///
/// Compared on segment boundaries rather than as a bare substring: a base of
/// `.../v1/mock_models` ends with the characters `models` without ending with
/// the segment `models`, and must still have `/models` appended.
fn ends_with_path_segment(base: &str, path: &str) -> bool {
    base == path || base.ends_with(&format!("/{path}"))
}

pub fn build_upstream_url(base_url: &str, path: &str) -> String {
    let path = path.trim_start_matches('/');
    if let Some((base, query)) = base_url.split_once('?') {
        let base = base.trim_end_matches('/');
        if ends_with_path_segment(base, path) {
            format!("{base}?{query}")
        } else {
            format!("{base}/{path}?{query}")
        }
    } else {
        let base = base_url.trim_end_matches('/');
        if ends_with_path_segment(base, path) {
            base.to_string()
        } else {
            format!("{base}/{path}")
        }
    }
}

/// Strip credentials from a URL so it can be logged.
///
/// Some providers carry the key in the query string (`?key=…`), and userinfo
/// (`https://user:pass@host`) is a standard place for one. Neither belongs in a
/// log line.
pub fn redact_url(url: &str) -> String {
    let (base, had_query) = match url.split_once('?') {
        Some((base, _)) => (base, true),
        None => (url, false),
    };

    let mut out = match base.split_once("://") {
        Some((scheme, rest)) => {
            let authority_end = rest.find('/').unwrap_or(rest.len());
            match rest[..authority_end].rfind('@') {
                Some(at) => format!("{scheme}://<redacted>@{}", &rest[at + 1..]),
                None => base.to_string(),
            }
        }
        None => base.to_string(),
    };

    if had_query {
        out.push_str("?<redacted>");
    }
    out
}

pub fn expand_path(p: &Path) -> PathBuf {
    if let Some(str_val) = p.to_str() {
        expand_tilde(str_val)
    } else {
        p.to_path_buf()
    }
}

pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home);
        }
    } else if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.bind, "127.0.0.1:8787");
        assert_eq!(config.server.max_body_bytes, 32 * 1024 * 1024);
        assert_eq!(config.upstream.connect_timeout_secs, 10);
        assert_eq!(config.resume.mode, ContinuationMode::Instructed);
        assert_eq!(config.resume.max_attempts, 6);
        assert_eq!(config.resume.dedup_window_chars, 240);
        assert_eq!(config.resume.max_no_progress_attempts, 3);
        assert!(config.resume.enabled);
        assert!(config.tools.buffer_tool_calls);
    }

    #[test]
    fn test_parse_toml() {
        let toml_str = r#"
        [server]
        bind = "0.0.0.0:9000"

        [resume]
        mode = "instructed"
        max_attempts = 3
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.server.bind, "0.0.0.0:9000");
        assert_eq!(config.resume.mode, ContinuationMode::Instructed);
        assert_eq!(config.resume.max_attempts, 3);
        assert_eq!(
            config.upstream.base_url,
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
    }

    #[test]
    fn test_build_upstream_url() {
        assert_eq!(
            build_upstream_url("https://api.openai.com/v1", "chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            build_upstream_url("https://api.openai.com/v1/", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            build_upstream_url(
                "http://127.0.0.1:8080/v1/chat/completions?cut_after=5",
                "chat/completions"
            ),
            "http://127.0.0.1:8080/v1/chat/completions?cut_after=5"
        );
        assert_eq!(
            build_upstream_url("http://127.0.0.1:8080/v1?custom=1", "models"),
            "http://127.0.0.1:8080/v1/models?custom=1"
        );
    }

    /// A base whose final segment merely *ends with* the path must still have
    /// the path appended — `ends_with` on bare substrings collides here.
    #[test]
    fn test_build_upstream_url_respects_segment_boundaries() {
        assert_eq!(
            build_upstream_url("http://h/v1/mock_models", "models"),
            "http://h/v1/mock_models/models"
        );
        assert_eq!(
            build_upstream_url("http://h/v1/xmodels?k=1", "models"),
            "http://h/v1/xmodels/models?k=1"
        );
        assert_eq!(
            build_upstream_url("http://h/v1/my_chat/completions", "chat/completions"),
            "http://h/v1/my_chat/completions/chat/completions"
        );
        // A genuine trailing segment is still detected and left alone.
        assert_eq!(
            build_upstream_url("http://h/v1/models", "models"),
            "http://h/v1/models"
        );
        assert_eq!(
            build_upstream_url("http://h/v1/chat/completions", "chat/completions"),
            "http://h/v1/chat/completions"
        );
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse_bool("true"), Ok(true));
        assert_eq!(parse_bool("1"), Ok(true));
        assert_eq!(parse_bool("yes"), Ok(true));
        assert_eq!(parse_bool("on"), Ok(true));
        assert_eq!(parse_bool("false"), Ok(false));
        assert_eq!(parse_bool("0"), Ok(false));
        assert_eq!(parse_bool("no"), Ok(false));
        assert_eq!(parse_bool("off"), Ok(false));
        assert!(parse_bool("invalid").is_err());
    }

    #[test]
    fn test_api_key_resolution() {
        let mut config = Config::default();
        config.upstream.api_key = Some("direct_key".to_string());
        assert_eq!(
            config.get_upstream_api_key(),
            Some("direct_key".to_string())
        );

        config.upstream.api_key = None;
        config.upstream.api_key_env = "TEST_PROXY_NONEXISTENT_KEY_VAR".to_string();
        assert_eq!(config.get_upstream_api_key(), None);
    }

    /// Environment variables are process-global; every test that mutates them must
    /// serialise through this lock so parallel test threads cannot observe each
    /// other's overrides.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Guard that removes the variables it was given when it drops, so a failing
    /// assertion cannot leak overrides into the rest of the test binary.
    struct EnvVars<'a> {
        keys: Vec<&'a str>,
        _guard: std::sync::MutexGuard<'a, ()>,
    }

    impl<'a> EnvVars<'a> {
        fn set(pairs: &[(&'a str, &str)]) -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut keys = Vec::with_capacity(pairs.len());
            for (key, val) in pairs {
                env::set_var(key, val);
                keys.push(*key);
            }
            Self {
                keys,
                _guard: guard,
            }
        }
    }

    impl Drop for EnvVars<'_> {
        fn drop(&mut self) {
            for key in &self.keys {
                env::remove_var(key);
            }
        }
    }

    #[test]
    fn test_env_overrides() {
        let _env = EnvVars::set(&[
            ("LLM_PROXY_SERVER_BIND", "127.0.0.1:9999"),
            ("LLM_PROXY_RESUME_MAX_ATTEMPTS", "10"),
            ("LLM_PROXY_RESUME_MODE", "instructed"),
            ("LLM_PROXY_RESUME_ENABLED", "false"),
        ]);

        let mut config = Config::default();
        config.apply_env_overrides().unwrap();

        assert_eq!(config.server.bind, "127.0.0.1:9999");
        assert_eq!(config.resume.max_attempts, 10);
        assert_eq!(config.resume.mode, ContinuationMode::Instructed);
        assert!(!config.resume.enabled);
    }

    /// §4: "Every field is overridable by environment variable
    /// (`LLM_PROXY_<SECTION>_<KEY>`)." This exercises the complete matrix so a
    /// newly added config field cannot silently ship without an override.
    #[test]
    fn test_env_override_matrix_covers_every_field() {
        let _env = EnvVars::set(&[
            ("LLM_PROXY_SERVER_BIND", "0.0.0.0:1234"),
            ("LLM_PROXY_SERVER_MAX_BODY_BYTES", "1048576"),
            ("LLM_PROXY_UPSTREAM_BASE_URL", "http://example.invalid/v9"),
            ("LLM_PROXY_UPSTREAM_API_KEY_ENV", "SOME_OTHER_KEY_VAR"),
            ("LLM_PROXY_UPSTREAM_API_KEY", "direct-key-from-env"),
            ("LLM_PROXY_UPSTREAM_CONNECT_TIMEOUT_SECS", "42"),
            ("LLM_PROXY_RESUME_ENABLED", "off"),
            ("LLM_PROXY_RESUME_MODE", "instructed"),
            ("LLM_PROXY_RESUME_MAX_ATTEMPTS", "11"),
            ("LLM_PROXY_RESUME_MAX_WALL_CLOCK_SECS", "1234"),
            ("LLM_PROXY_RESUME_OVERLOADED_MAX_WALL_CLOCK_SECS", "4321"),
            ("LLM_PROXY_RESUME_BACKOFF_INITIAL_MS", "123"),
            ("LLM_PROXY_RESUME_BACKOFF_MAX_MS", "45678"),
            ("LLM_PROXY_RESUME_BACKOFF_MULTIPLIER", "3.5"),
            ("LLM_PROXY_RESUME_BACKOFF_JITTER", "0.125"),
            ("LLM_PROXY_RESUME_DEDUP_WINDOW_CHARS", "999"),
            ("LLM_PROXY_RESUME_FIRST_CHUNK_TIMEOUT_SECS", "77"),
            ("LLM_PROXY_RESUME_IDLE_TIMEOUT_SECS", "13"),
            ("LLM_PROXY_RESUME_MAX_NO_PROGRESS_ATTEMPTS", "7"),
            ("LLM_PROXY_RESUME_RESUME_ON_LENGTH_FINISH", "yes"),
            ("LLM_PROXY_TOOLS_BUFFER_TOOL_CALLS", "0"),
            ("LLM_PROXY_TOOLS_DISABLE_RESUME_FOR_TOOL_RESPONSES", "1"),
            ("LLM_PROXY_LOG_LEVEL", "trace"),
            ("LLM_PROXY_LOG_LOG_BODIES", "true"),
        ]);

        let mut config = Config::default();
        config.apply_env_overrides().unwrap();

        assert_eq!(config.server.bind, "0.0.0.0:1234");
        assert_eq!(config.server.max_body_bytes, 1_048_576);

        assert_eq!(config.upstream.base_url, "http://example.invalid/v9");
        assert_eq!(config.upstream.api_key_env, "SOME_OTHER_KEY_VAR");
        assert_eq!(
            config.upstream.api_key.as_deref(),
            Some("direct-key-from-env")
        );
        assert_eq!(config.upstream.connect_timeout_secs, 42);

        assert!(!config.resume.enabled);
        assert_eq!(config.resume.mode, ContinuationMode::Instructed);
        assert_eq!(config.resume.max_attempts, 11);
        assert_eq!(config.resume.max_wall_clock_secs, 1234);
        assert_eq!(config.resume.overloaded_max_wall_clock_secs, 4321);
        assert_eq!(config.resume.backoff_initial_ms, 123);
        assert_eq!(config.resume.backoff_max_ms, 45678);
        assert_eq!(config.resume.backoff_multiplier, 3.5);
        assert_eq!(config.resume.backoff_jitter, 0.125);
        assert_eq!(config.resume.dedup_window_chars, 999);
        assert_eq!(config.resume.first_chunk_timeout_secs, 77);
        assert_eq!(config.resume.idle_timeout_secs, 13);
        assert_eq!(config.resume.max_no_progress_attempts, 7);
        assert!(config.resume.resume_on_length_finish);

        assert!(!config.tools.buffer_tool_calls);
        assert!(config.tools.disable_resume_for_tool_responses);

        assert_eq!(config.log.level, "trace");
        assert!(config.log.log_bodies);

        // The direct key override wins over the (now bogus) key env var name.
        assert_eq!(
            config.get_upstream_api_key(),
            Some("direct-key-from-env".to_string())
        );
    }

    #[test]
    fn test_env_override_rejects_unparseable_values() {
        for (key, val) in [
            ("LLM_PROXY_SERVER_MAX_BODY_BYTES", "big"),
            ("LLM_PROXY_UPSTREAM_CONNECT_TIMEOUT_SECS", "ten"),
            ("LLM_PROXY_RESUME_ENABLED", "maybe"),
            ("LLM_PROXY_RESUME_MODE", "verbatim"),
            ("LLM_PROXY_RESUME_MAX_ATTEMPTS", "-1"),
            ("LLM_PROXY_RESUME_MAX_WALL_CLOCK_SECS", "forever"),
            ("LLM_PROXY_RESUME_OVERLOADED_MAX_WALL_CLOCK_SECS", "ages"),
            ("LLM_PROXY_RESUME_BACKOFF_INITIAL_MS", "1.5"),
            ("LLM_PROXY_RESUME_BACKOFF_MAX_MS", "lots"),
            ("LLM_PROXY_RESUME_BACKOFF_MULTIPLIER", "double"),
            ("LLM_PROXY_RESUME_BACKOFF_JITTER", "some"),
            ("LLM_PROXY_RESUME_DEDUP_WINDOW_CHARS", "-5"),
            ("LLM_PROXY_RESUME_FIRST_CHUNK_TIMEOUT_SECS", "soon"),
            ("LLM_PROXY_RESUME_IDLE_TIMEOUT_SECS", "soon"),
            ("LLM_PROXY_RESUME_MAX_NO_PROGRESS_ATTEMPTS", "thrice"),
            ("LLM_PROXY_RESUME_RESUME_ON_LENGTH_FINISH", "sometimes"),
            ("LLM_PROXY_TOOLS_BUFFER_TOOL_CALLS", "perhaps"),
            (
                "LLM_PROXY_TOOLS_DISABLE_RESUME_FOR_TOOL_RESPONSES",
                "perhaps",
            ),
        ] {
            let _env = EnvVars::set(&[(key, val)]);
            let mut config = Config::default();
            let err = config
                .apply_env_overrides()
                .expect_err("{key}={val} should be rejected");
            match err {
                ConfigError::EnvOverride { key: got, .. } => assert_eq!(got, key),
                other => panic!("{key} produced unexpected error: {other}"),
            }
        }
    }

    #[test]
    fn test_api_key_env_var_is_read_when_no_direct_key() {
        let _env = EnvVars::set(&[
            (
                "LLM_PROXY_UPSTREAM_API_KEY_ENV",
                "LLM_PROXY_TEST_KEY_SOURCE",
            ),
            ("LLM_PROXY_TEST_KEY_SOURCE", "key-from-named-var"),
        ]);

        let mut config = Config::default();
        config.apply_env_overrides().unwrap();

        assert_eq!(
            config.get_upstream_api_key(),
            Some("key-from-named-var".to_string())
        );

        // A whitespace-only value is treated as absent, not as an empty key.
        env::set_var("LLM_PROXY_TEST_KEY_SOURCE", "   ");
        assert_eq!(config.get_upstream_api_key(), None);
    }

    /// A typo in a config key must fail loudly: silently falling back to the
    /// default for `max_attempts` or `base_url` is a security and cost problem.
    #[test]
    fn test_unknown_toml_keys_are_rejected() {
        let err = toml::from_str::<Config>(
            r#"
            [resume]
            max_attemps = 3
            "#,
        )
        .expect_err("a misspelled key should be rejected");
        assert!(err.to_string().contains("max_attemps"), "{err}");

        assert!(toml::from_str::<Config>(
            r#"
            [nonsense]
            whatever = 1
            "#,
        )
        .is_err());
    }

    /// A path named through `$LLM_PROXY_CONFIG` was chosen deliberately, so a
    /// typo in it must error rather than quietly loading defaults — which would
    /// route traffic to the built-in Gemini endpoint.
    #[test]
    fn test_missing_env_config_path_is_an_error() {
        let _env = EnvVars::set(&[(
            "LLM_PROXY_CONFIG",
            "/nonexistent/llm-proxy/definitely-not-here.toml",
        )]);
        match Config::load(None) {
            Err(ConfigError::Io { path, .. }) => {
                assert!(path.to_string_lossy().contains("definitely-not-here"));
            }
            Ok(_) => panic!("a missing env-selected config must not fall back to defaults"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_api_key_is_trimmed() {
        let mut config = Config::default();
        // A key read from a file or here-doc carries a trailing newline, which
        // is not a legal header value.
        config.upstream.api_key = Some("  sk-test-key\n".to_string());
        assert_eq!(
            config.get_upstream_api_key(),
            Some("sk-test-key".to_string())
        );

        config.upstream.api_key = Some("   ".to_string());
        config.upstream.api_key_env = "LLM_PROXY_NONEXISTENT_KEY_VAR".to_string();
        assert_eq!(config.get_upstream_api_key(), None);
    }

    #[test]
    fn test_redact_url_strips_credentials() {
        assert_eq!(
            redact_url("https://generativelanguage.googleapis.com/v1beta/openai"),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            redact_url("https://host/v1?key=secret-value"),
            "https://host/v1?<redacted>"
        );
        assert_eq!(
            redact_url("https://user:pass@host/v1"),
            "https://<redacted>@host/v1"
        );
        assert_eq!(
            redact_url("https://user:pass@host/v1?key=secret"),
            "https://<redacted>@host/v1?<redacted>"
        );
        // An '@' in the path is not userinfo and must be left alone.
        assert_eq!(redact_url("http://host/v1/a@b"), "http://host/v1/a@b");
    }

    /// Values that parse but cannot work must be refused at startup rather than
    /// producing nonsense at request time.
    #[test]
    fn test_validate_rejects_unusable_values() {
        type Mutate = Box<dyn Fn(&mut Config)>;
        let cases: Vec<(&str, Mutate)> = vec![
            (
                "max_attempts",
                Box::new(|c: &mut Config| c.resume.max_attempts = 0),
            ),
            (
                "max_concurrent_requests",
                Box::new(|c: &mut Config| c.server.max_concurrent_requests = 0),
            ),
            (
                "dedup_window_chars",
                Box::new(|c: &mut Config| c.resume.dedup_window_chars = 0),
            ),
            (
                "max_body_bytes",
                Box::new(|c: &mut Config| c.server.max_body_bytes = 0),
            ),
            (
                "idle_timeout_secs",
                Box::new(|c: &mut Config| c.resume.idle_timeout_secs = 0),
            ),
            (
                "max_wall_clock_secs",
                Box::new(|c: &mut Config| c.resume.max_wall_clock_secs = 999_999_999),
            ),
            (
                "backoff_multiplier",
                Box::new(|c: &mut Config| c.resume.backoff_multiplier = f64::NAN),
            ),
            (
                "backoff_multiplier",
                Box::new(|c: &mut Config| c.resume.backoff_multiplier = 0.5),
            ),
            (
                "backoff_jitter",
                Box::new(|c: &mut Config| c.resume.backoff_jitter = f64::INFINITY),
            ),
            (
                "backoff_jitter",
                Box::new(|c: &mut Config| c.resume.backoff_jitter = 2.0),
            ),
            (
                "backoff_max_ms",
                Box::new(|c: &mut Config| {
                    c.resume.backoff_initial_ms = 5_000;
                    c.resume.backoff_max_ms = 100;
                }),
            ),
            (
                "base_url",
                Box::new(|c: &mut Config| c.upstream.base_url = "not a url".to_string()),
            ),
            (
                "base_url scheme",
                Box::new(|c: &mut Config| c.upstream.base_url = "ftp://host/v1".to_string()),
            ),
            (
                "api key header",
                Box::new(|c: &mut Config| c.upstream.api_key = Some("bad\nkey".to_string())),
            ),
        ];

        for (label, mutate) in cases {
            let mut config = Config::default();
            mutate(&mut config);
            assert!(
                matches!(config.validate(), Err(ConfigError::Validation(_))),
                "{label} should have been rejected"
            );
        }

        // The shipped defaults must of course pass.
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn test_config_path_honours_llm_proxy_config_env() {
        let _env = EnvVars::set(&[("LLM_PROXY_CONFIG", "/tmp/some-proxy-config.toml")]);
        assert_eq!(
            Config::default_config_path(),
            PathBuf::from("/tmp/some-proxy-config.toml")
        );
    }
}
