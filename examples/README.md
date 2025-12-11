# BorgLLM Examples

This directory contains examples organized by integration type:

## Directory Structure

```
examples/
├── openai/          # Core OpenAI SDK examples (no extra dependencies)
└── langchain/       # LangChain integration examples (requires borgllm[langchain])
```

## Installation

### Core (OpenAI SDK)
```bash
pip install borgllm
```

### With LangChain Support
```bash
pip install borgllm[langchain]
```

---

## OpenAI Examples (`examples/openai/`)

These examples work with the core `borgllm` install and demonstrate the `BorgOpenAI` and `BorgAsyncOpenAI` clients.

### [basic_usage/main.py](openai/basic_usage/main.py)
- **Purpose**: Simplest example using BorgOpenAI with built-in providers
- **Key Features**: `BorgOpenAI`, `chat.completions.create()`, multi-turn conversations
- **Requires `.env`**: Yes (`OPENAI_API_KEY`)

### [basic_usage/async_example.py](openai/basic_usage/async_example.py)
- **Purpose**: Async client usage with concurrent requests and streaming
- **Key Features**: `BorgAsyncOpenAI`, `asyncio.gather()`, async streaming
- **Requires `.env`**: Yes (`OPENAI_API_KEY`)

### [custom_provider/main.py](openai/custom_provider/main.py)
- **Purpose**: Configure custom providers via `borg.yaml` (Ollama, LM Studio, etc.)
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes

### [init_from_dict/main.py](openai/init_from_dict/main.py)
- **Purpose**: Programmatic configuration via Python dictionary
- **Uses `borg.yaml`**: No
- **Requires `.env`**: Yes

### [virtual_provider/main.py](openai/virtual_provider/main.py)
- **Purpose**: Virtual providers with automatic fallback between upstreams
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes (`GROQ_API_KEY`, `CEREBRAS_API_KEY`)

### [multiple_api_keys/main.py](openai/multiple_api_keys/main.py)
- **Purpose**: Round-robin API key rotation
- **Requires `.env`**: Yes (`OPENAI_API_KEYS` comma-separated)

### [provider_cooldown/main.py](openai/provider_cooldown/main.py)
- **Purpose**: 429 error handling and cooldown management
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes (`GROQ_API_KEY`, `CEREBRAS_API_KEY`)

### [configurable_cooldown_timeout/main.py](openai/configurable_cooldown_timeout/main.py)
- **Purpose**: Custom cooldown and timeout configuration (global, per-provider, per-model)
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Optional

### [streaming/main.py](openai/streaming/main.py)
- **Purpose**: Streaming responses with sync and async clients
- **Key Features**: Sync streaming, async streaming, virtual provider streaming
- **Requires `.env`**: Yes (`OPENAI_API_KEY`)

---

## LangChain Examples (`examples/langchain/`)

These examples require `pip install borgllm[langchain]` and demonstrate the `create_llm` function and `BorgLLMLangChainClient`.

### [basic_usage/main.py](langchain/basic_usage/main.py)
- **Purpose**: Simplest LangChain example using `create_llm` with built-in providers
- **Key Features**: `create_llm`, `invoke` with text and message lists
- **Requires `.env`**: Yes (`OPENAI_API_KEY`)

### [custom_provider/main.py](langchain/custom_provider/main.py)
- **Purpose**: Configure custom providers via `borg.yaml`
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes

### [init_from_dict/main.py](langchain/init_from_dict/main.py)
- **Purpose**: Programmatic configuration via Python dictionary
- **Uses `borg.yaml`**: No
- **Requires `.env`**: Yes

### [default_virtual_provider/main.py](langchain/default_virtual_provider/main.py)
- **Purpose**: Virtual providers with automatic fallback
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes (`GROQ_API_KEY`, `CEREBRAS_API_KEY`)

### [multiple_api_keys/main.py](langchain/multiple_api_keys/main.py)
- **Purpose**: Round-robin API key rotation
- **Requires `.env`**: Yes (`OPENAI_API_KEYS` comma-separated)

### [virtual_provider_token_approx/main.py](langchain/virtual_provider_token_approx/main.py)
- **Purpose**: Dynamic provider selection based on token count
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes

### [provider_cooldown/main.py](langchain/provider_cooldown/main.py)
- **Purpose**: 429 error handling and cooldown management
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Yes

### [configurable_cooldown_timeout/main.py](langchain/configurable_cooldown_timeout/main.py)
- **Purpose**: Custom cooldown and timeout configuration
- **Uses `borg.yaml`**: Yes
- **Requires `.env`**: Optional

### [virtual_provider_await_cooldown/main.py](langchain/virtual_provider_await_cooldown/main.py)
- **Purpose**: Await cooldown expiration before returning provider
- **Key Features**: `allow_await_cooldown`, `timeout`
- **Requires `.env`**: Yes

### [virtual_provider_no_await/main.py](langchain/virtual_provider_no_await/main.py)
- **Purpose**: Immediate failure when no providers available
- **Key Features**: `allow_await_cooldown=False`
- **Requires `.env`**: Yes

---

## Running Examples

1. Install dependencies:
   ```bash
   # For OpenAI examples
   pip install borgllm
   
   # For LangChain examples
   pip install borgllm[langchain]
   ```

2. Create a `.env` file in the example directory with required API keys

3. Run the example:
   ```bash
   cd examples/openai
   python main.py
   ``` 