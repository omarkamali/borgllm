# BorgLLM Examples

This directory contains various examples demonstrating how to use BorgLLM in different scenarios. Each example is self-contained within its own folder and includes a `main.py` script.

To run any of these examples:
1. Ensure you have `uv` installed (`pip install uv`).
2. Run `uv pip install` in the project root to install dependencies.
3. For examples that require API keys, create a `.env` file in the *example's directory* (next to `main.py`) with the necessary keys (e.g., `OPENAI_API_KEY=your_key_here`).
4. Navigate to the specific example's directory (e.g., `cd examples/langchain_simple`) and run the script: `python3 main.py`.

---

### 1. [Basic Usage (`basic_usage/main.py`)](basic_usage/main.py)
- **Purpose**: The simplest example demonstrating how to create and use a LangChain LLM client with BorgLLM's `create_llm` function. This example uses built-in providers and does **not** require a `borg.yaml` file.
- **Key Functionality**: Basic `create_llm` usage, `invoke` with text and message lists.
- **Requires `.env`**: Yes (for `OPENAI_API_KEY`)

### 2. [Custom Provider (`custom_provider/main.py`)](custom_provider/main.py)
- **Purpose**: Demonstrates how to configure and retrieve a custom LLM provider defined in a `borg.yaml` file.
- **Uses `borg.yaml`**: Yes (`custom_provider/borg.yaml`)
- **Key Functionality**: Initializing BorgLLM from a file, getting a custom provider.
- **Requires `.env`**: Yes (for `OPENAI_API_KEY` or equivalent key for the custom provider)


### 3. [Initialize from Dictionary (`init_from_dict/main.py`)](init_from_dict/main.py)
- **Purpose**: Shows how to initialize BorgLLM programmatically by providing configuration data as a Python dictionary, instead of loading from a `borg.yaml` file.
- **Uses `borg.yaml`**: No (configuration is in `main.py`)
- **Key Functionality**: Programmatic configuration initialization.
- **Requires `.env`**: Yes (for `OPENAI_API_KEY` or equivalent dummy key for the custom provider)

### 4. [Default Virtual Provider (`default_virtual_provider/main.py`)](default_virtual_provider/main.py)
- **Purpose**: Illustrates the use of a "virtual" provider defined in `borg.yaml`, which allows for automatic fallback between multiple upstream LLM providers.
- **Uses `borg.yaml`**: Yes (`default_virtual_provider/borg.yaml`)
- **Key Functionality**: Retrieving the default virtual provider, understanding provider fallback.
- **Requires `.env`**: Yes (for `GROQ_API_KEY`, `CEREBRAS_API_KEY` or equivalent dummy keys for custom providers)

### 5. [Multiple API Keys (`multiple_api_keys/main.py`)](multiple_api_keys/main.py)
- **Purpose**: Shows how to configure and use multiple API keys for a built-in provider, demonstrating the automatic round-robin rotation of keys.
- **Uses `borg.yaml`**: No
- **Key Functionality**: API key rotation for built-in providers via environment variables.
- **Requires `.env`**: Yes (for `OPENAI_API_KEYS` with comma-separated keys)
    
### 6. [Virtual Provider Token Approximation (`virtual_provider_token_approx/main.py`)](virtual_provider_token_approx/main.py)
- **Purpose**: Shows how BorgLLM's virtual providers can select the best upstream provider based on the approximate number of tokens required for a task.
- **Uses `borg.yaml`**: Yes (`virtual_provider_token_approx/borg.yaml`)
- **Key Functionality**: Dynamic provider selection based on `approximate_tokens`.
- **Requires `.env`**: Yes (for `GROQ_API_KEY`, `CEREBRAS_API_KEY` or equivalent keys for custom providers)

### 7. [Provider Cooldown (`provider_cooldown/main.py`)](provider_cooldown/main.py)
- **Purpose**: Demonstrates BorgLLM's built-in 429 (Too Many Requests) error handling, including signaling a cooldown for a provider and how virtual providers intelligently avoid blocked upstreams.
- **Uses `borg.yaml`**: Yes (`provider_cooldown/borg.yaml`)
- **Key Functionality**: `signal_429`, automatic provider avoidance, cooldown expiration.
- **Requires `.env`**: Yes (for `GROQ_API_KEY`, `CEREBRAS_API_KEY` or equivalent keys for custom providers)

### 8. [Configurable Cooldown and Timeout (`configurable_cooldown_timeout/main.py`)](configurable_cooldown_timeout/main.py)
- **Purpose**: Shows how to configure custom cooldown periods and request timeouts globally, for specific providers, or for specific `provider:model` combinations.
- **Uses `borg.yaml`**: Yes (`configurable_cooldown_timeout/borg.yaml`)
- **Key Functionality**: `cooldown` and `timeout` parameters in `create_llm()`, `set_cooldown_config()`, `set_timeout_config()`, and how these affect provider selection and rate limit handling.
- **Requires `.env`**: Optional (for `OPENAI_API_KEY` if running examples that make actual API calls; dummy providers are used if not present).

### 9. [Virtual Provider Await Cooldown (`virtual_provider_await_cooldown/main.py`)](virtual_provider_await_cooldown/main.py)
- **Purpose**: Shows how a virtual provider can optionally await a blocked upstream's cooldown period to expire before returning a provider. Includes a scenario for timeout.
- **Uses `borg.yaml`**: No (uses programmatic configuration for a temporary virtual provider)
- **Key Functionality**: `allow_await_cooldown`, `timeout` for awaiting.
- **Requires `.env`**: Yes (for `OPENAI_API_KEY` or equivalent key for the custom provider)

### 10. [Virtual Provider No Await (`virtual_provider_no_await/main.py`)](virtual_provider_no_await/main.py)
- **Purpose**: Demonstrates the behavior of a virtual provider when it is explicitly configured *not* to await an upstream's cooldown, resulting in immediate failure if no other suitable provider is available.
- **Uses `borg.yaml`**: No (uses programmatic configuration for a temporary virtual provider)
- **Key Functionality**: `allow_await_cooldown=False`.
- **Requires `.env`**: Yes (for `OPENAI_API_KEY` or equivalent key for the custom provider) 