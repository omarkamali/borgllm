import os
import yaml
import time
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
import threading
import logging
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
    ConfigDict,
)


# Initialize logger
logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()


# Helper function to parse API keys from various formats
def _parse_api_keys(api_key_input: Union[str, List[str]]) -> List[str]:
    """Parse API keys from various input formats."""
    if isinstance(api_key_input, list):
        return [key.strip() for key in api_key_input if key.strip()]
    elif isinstance(api_key_input, str):
        if "," in api_key_input:
            return [key.strip() for key in api_key_input.split(",") if key.strip()]
        else:
            return [api_key_input.strip()] if api_key_input.strip() else []
    else:
        return []


# Define built-in providers with their base URLs and corresponding environment variable prefixes
BUILTIN_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "max_tokens": 4096,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-sonnet-20240620",
        "max_tokens": 4096,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "mistralai/mistral-7b-instruct",
        "max_tokens": 4096,
    },
    "togetherai": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "max_tokens": 4096,
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "api_key_env": "PERPLEXITY_API_KEY",
        "default_model": "llama-3-sonar-small-32k-online",
        "max_tokens": 32768,
    },
    "mistralai": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
        "max_tokens": 32768,
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",  # Note: Fireworks specific endpoint often ends with /v1 or /v1/chat/completions
        "api_key_env": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "max_tokens": 32768,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama3-8b-8192",
        "max_tokens": 32768,
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1",
        "api_key_env": "DEEPINFRA_API_KEY",
        "default_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_tokens": 32768,
    },
    "anyscale": {
        "base_url": "https://api.endpoints.anyscale.com/v1",
        "api_key_env": "ANYSCALE_API_KEY",
        "default_model": "meta-llama/Llama-2-7b-chat-hf",
        "max_tokens": 4096,
    },
    "novita": {
        "base_url": "https://api.novita.ai/v1",  # Based on common OpenAI-compatible patterns, though documentation might specify /v1/chat/completions
        "api_key_env": "NOVITA_API_KEY",
        "default_model": "llama2-7b-chat",
        "max_tokens": 8192,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "default_model": "llama3.1-8b",
        "max_tokens": 2048,
    },
    "featherless": {
        "base_url": "https://api.featherless.ai/v1",
        "api_key_env": "FEATHERLESS_API_KEY",
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_tokens": 8192,
    },
    "cohere": {
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "api_key_env": "COHERE_API_KEY",
        "default_model": "command-r-plus",
        "max_tokens": 131072,
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "max_tokens": 32768,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "max_tokens": 32768,
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
        "max_tokens": 32768,
    },
    "moonshot": {
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "kimi-k2-0711-preview",
        "max_tokens": 131072,
    },
    "omneity": {
        "base_url": "https://api.sawalni.com/v1",
        "api_key_env": "OMNEITY_API_KEY",
        "default_model": "sawalni-beta",
        "max_tokens": 8192,
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
        "default_model": "zai/glm-4.6",
        "max_tokens": 200000,
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "default_model": "minimax-m2",
        "max_tokens": 128000,
    },

}


class LLMProviderConfig(BaseModel):
    name: str
    base_url: HttpUrl
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = Field(..., gt=0)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **data):
        super().__init__(**data)
        if not hasattr(self, "_api_keys") or not self._api_keys:
            self._api_keys: List[str] = [self.api_key] if self.api_key else []
        if not hasattr(self, "_current_key_index"):
            self._current_key_index: int = 0

    @model_validator(mode="before")
    @classmethod
    def validate_api_key(cls, data):
        if isinstance(data, dict):
            api_keys_value = data.get("api_keys")
            api_key_value = data.get("api_key")

            if api_keys_value is not None:
                api_keys = _parse_api_keys(api_keys_value)
                if api_keys:
                    data["api_key"] = api_keys[0]
                    data["_api_keys"] = api_keys
            elif api_key_value is not None:
                api_keys = _parse_api_keys(api_key_value)
                data["api_key"] = api_keys[0] if api_keys else api_key_value
                data["_api_keys"] = api_keys

            if "api_keys" in data:
                del data["api_keys"]
        return data

    def get_next_api_key(self) -> str:
        """Get the next API key in round-robin fashion."""
        if not self._api_keys:
            return self.api_key

        current_key = self._api_keys[self._current_key_index]
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        self.api_key = current_key
        return current_key

    def set_api_keys(self, api_keys: List[str]):
        """Set the list of API keys for round-robin."""
        if api_keys:
            self._api_keys = api_keys
            self._current_key_index = 0
            self.api_key = api_keys[0]

    def has_multiple_keys(self) -> bool:
        """Check if this provider has multiple API keys."""
        return len(self._api_keys) > 1


# Global cache for built-in providers to share state across all BorgLLM instances
_GLOBAL_BUILTIN_PROVIDERS: Dict[str, LLMProviderConfig] = {}
_GLOBAL_BUILTIN_LOCK = threading.Lock()


class VirtualLLMProviderConfig(BaseModel):
    name: str
    upstreams: List[Dict[str, str]]


class LLMConfig(BaseModel):
    providers: List[LLMProviderConfig]
    virtual: Optional[List[VirtualLLMProviderConfig]] = None
    default_model: Optional[str] = None


class BorgLLM:
    _instance = None
    _config_initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BorgLLM, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: str = "borg.yaml",
        initial_config_data: Optional[Dict] = None,
        _force_reinitialize: bool = False,
    ):
        if not hasattr(self, "_default_provider_name") or _force_reinitialize:
            self._default_provider_name: Optional[str] = None
        if not hasattr(self, "_config") or _force_reinitialize:
            self._config: Optional[LLMConfig] = None
        if not hasattr(self, "_real_providers") or _force_reinitialize:
            self._real_providers: Dict[str, LLMProviderConfig] = {}
        if not hasattr(self, "_virtual_providers") or _force_reinitialize:
            self._virtual_providers: Dict[str, VirtualLLMProviderConfig] = {}
        if not hasattr(self, "_unusable_providers") or _force_reinitialize:
            self._unusable_providers: Dict[str, float] = {}
        if not hasattr(self, "_virtual_provider_last_index") or _force_reinitialize:
            self._virtual_provider_last_index: Dict[str, int] = {}
        if not hasattr(self, "_cooldown_config") or _force_reinitialize:
            self._cooldown_config: Union[int, Dict[str, int]] = 60  # Default 60 seconds
        if not hasattr(self, "_timeout_config") or _force_reinitialize:
            self._timeout_config: Optional[Union[float, Dict[str, float]]] = None

        if BorgLLM._config_initialized and not _force_reinitialize:
            logger.debug("BorgLLM already initialized, skipping re-initialization.")
            return

        with self._lock:
            if not BorgLLM._config_initialized or _force_reinitialize:
                self.config_path = config_path
                self._config_paths = self._get_config_paths(config_path)
                self._config: Optional[LLMConfig] = None
                self._real_providers: Dict[str, LLMProviderConfig] = {}
                self._virtual_providers: Dict[str, VirtualLLMProviderConfig] = {}
                self._default_provider_name: Optional[str] = None

                if initial_config_data:
                    interpolated_data = self._interpolate_env_variables(
                        initial_config_data
                    )
                    try:
                        self._config = LLMConfig(**interpolated_data["llm"])
                    except ValidationError as e:
                        logger.error(f"Initial configuration validation error: {e}")
                        raise

                self._load_config()
                self._populate_providers()
                self._add_builtin_providers()

                logger.info("\n--- BorgLLM Configuration Summary ---")
                if self._real_providers:
                    for provider_name, provider_config in self._real_providers.items():
                        masked_key = (
                            f"{provider_config.api_key[:4]}...{provider_config.api_key[-4:]}"
                            if provider_config.api_key
                            else "[NO KEY]"
                        )
                        num_keys = (
                            len(provider_config._api_keys)
                            if provider_config._api_keys
                            else 1
                        )
                        logger.info(
                            f"Provider '{provider_name}': {num_keys} API key(s) loaded (Current/First: {masked_key})"
                        )
                else:
                    logger.info(
                        "No LLM providers configured or found via environment variables."
                    )
                logger.info("-----------------------------------")

                self._virtual_provider_last_index: Dict[str, int] = {}
                self._unusable_providers: Dict[str, float] = {}

                BorgLLM._config_initialized = True

    @classmethod
    def get_instance(
        cls,
        config_path: str = "borg.yaml",
        initial_config_data: Optional[Dict] = None,
    ):
        """Get the singleton BorgLLM instance."""
        if cls._instance is None or not cls._config_initialized:
            cls(
                _force_reinitialize=True,
                config_path=config_path,
                initial_config_data=initial_config_data,
            )
        return cls._instance

    @property
    def config(self) -> Optional[LLMConfig]:
        """Public property to access the configuration."""
        return self._config

    @property
    def providers(self) -> Dict[str, LLMProviderConfig]:
        """Public property to access the real providers."""
        return self._real_providers

    def set_default_provider(self, provider_name: str):
        """Set the default LLM provider name for this BorgLLM instance."""
        is_builtin_reference = False
        if ":" in provider_name:
            provider_key = provider_name.split(":", 1)[0]
            if provider_key in BUILTIN_PROVIDERS:
                is_builtin_reference = True
        elif provider_name in BUILTIN_PROVIDERS:
            is_builtin_reference = True

        if (
            provider_name not in self._real_providers
            and provider_name not in self._virtual_providers
            and not is_builtin_reference
        ):
            raise ValueError(
                f"Provider '{provider_name}' not found. Cannot set as default."
            )

        with self._lock:
            self._default_provider_name = provider_name
            logger.info(
                f"Instance default LLM provider set to '{provider_name}' (overrides any config file default)."
            )

    def set_cooldown_config(self, cooldown: Union[int, Dict[str, int]]):
        """Set the cooldown configuration for providers.

        Args:
            cooldown: Either a global cooldown duration in seconds, or a dict mapping
                     provider names to specific cooldown durations. The dict can include
                     a "default" key for providers not explicitly listed.
        """
        with self._lock:
            self._cooldown_config = cooldown
            logger.info(f"Cooldown configuration updated: {cooldown}")

    def set_timeout_config(self, timeout: Union[float, Dict[str, float]]):
        """Set the timeout configuration for providers.

        Args:
            timeout: Either a global timeout duration in seconds, or a dict mapping
                    provider names to specific timeout durations. The dict can include
                    a "default" key for providers not explicitly listed.
        """
        with self._lock:
            self._timeout_config = timeout
            logger.info(f"Timeout configuration updated: {timeout}")

    def get_cooldown_duration(self, provider_name: str) -> int:
        """Get the cooldown duration for a specific provider.

        Args:
            provider_name: Name of the provider (may include model, e.g., "openai:gpt-4o")

        Returns:
            Cooldown duration in seconds
        """
        if isinstance(self._cooldown_config, int):
            return self._cooldown_config
        elif isinstance(self._cooldown_config, dict):
            # Check for exact provider match first
            if provider_name in self._cooldown_config:
                return self._cooldown_config[provider_name]
            # Check for provider key match (e.g., "openai" from "openai:gpt-4o")
            if ":" in provider_name:
                provider_key = provider_name.split(":", 1)[0]
                if provider_key in self._cooldown_config:
                    return self._cooldown_config[provider_key]
            # Fall back to default if specified
            if "default" in self._cooldown_config:
                return self._cooldown_config["default"]
            # Final fallback to global default
            return 60
        return 60

    def get_timeout_duration(self, provider_name: str) -> Optional[float]:
        """Get the timeout duration for a specific provider.

        Args:
            provider_name: Name of the provider (may include model, e.g., "openai:gpt-4o")

        Returns:
            Timeout duration in seconds, or None if no timeout is configured
        """
        if self._timeout_config is None:
            return None
        elif isinstance(self._timeout_config, (int, float)):
            return float(self._timeout_config)
        elif isinstance(self._timeout_config, dict):
            # Check for exact provider match first
            if provider_name in self._timeout_config:
                return self._timeout_config[provider_name]
            # Check for provider key match (e.g., "openai" from "openai:gpt-4o")
            if ":" in provider_name:
                provider_key = provider_name.split(":", 1)[0]
                if provider_key in self._timeout_config:
                    return self._timeout_config[provider_key]
            # Fall back to default if specified
            if "default" in self._timeout_config:
                return self._timeout_config["default"]
            # No timeout configured for this provider
            return None
        return None

    def _get_config_paths(self, base_path: str) -> List[str]:
        if base_path.endswith((".yaml", ".yml")):
            return [base_path]
        return [f"{base_path}.yaml", f"{base_path}.yml"]

    def _populate_providers(self):
        if not self._config:
            self._config = LLMConfig(providers=[], virtual=[], default_model=None)

        for provider in self._config.providers:
            self._real_providers[provider.name] = provider

        if self._config.virtual:
            for provider in self._config.virtual:
                self._virtual_providers[provider.name] = provider

        if self._config.default_model and not self._default_provider_name:
            self._default_provider_name = self._config.default_model

        # Static check: Verify all upstreams in virtual providers exist
        if self._config.virtual:
            for (
                virtual_provider_name,
                virtual_config,
            ) in self._virtual_providers.items():
                for upstream_info in virtual_config.upstreams:
                    upstream_name = upstream_info["name"]

                    # Check if it's a built-in provider reference (e.g., "cerebras:qwen-3-32b")
                    is_builtin_reference = False
                    if ":" in upstream_name:
                        provider_key = upstream_name.split(":", 1)[0]
                        if provider_key in BUILTIN_PROVIDERS:
                            is_builtin_reference = True
                    elif upstream_name in BUILTIN_PROVIDERS:
                        is_builtin_reference = True

                    if (
                        upstream_name not in self._real_providers
                        and upstream_name not in self._virtual_providers
                        and not is_builtin_reference
                    ):
                        raise ValueError(
                            f"Virtual provider '{virtual_provider_name}' references non-existent upstream '{upstream_name}'."
                        )

    def _add_builtin_providers(self):
        with _GLOBAL_BUILTIN_LOCK:
            for provider_name, settings in BUILTIN_PROVIDERS.items():
                # Support multiple API keys: check for both *_API_KEYS and *_API_KEY
                api_key_env = settings["api_key_env"]
                api_keys_env = api_key_env + "S"

                api_keys_value = os.getenv(api_keys_env)
                api_key_value = os.getenv(api_key_env)

                api_keys_list = []
                if api_keys_value:
                    api_keys_list = _parse_api_keys(api_keys_value)
                elif api_key_value:
                    api_keys_list = _parse_api_keys(api_key_value)

                if api_keys_list and provider_name not in self._real_providers:
                    try:
                        provider_data = {
                            "name": provider_name,
                            "base_url": settings["base_url"],
                            "model": settings["default_model"],
                            "api_key": api_keys_list[0],
                            "api_keys": api_keys_list,
                            "temperature": settings.get("temperature", 0.7),
                            "max_tokens": settings.get("max_tokens", 4096),
                        }

                        builtin_config = LLMProviderConfig(**provider_data)
                        _GLOBAL_BUILTIN_PROVIDERS[provider_name] = builtin_config
                        self._real_providers[provider_name] = builtin_config
                    except ValidationError as e:
                        logger.warning(
                            f"Error validating built-in provider {provider_name}: {e}"
                        )

    def _interpolate_env_variables(self, data):
        if isinstance(data, dict):
            return {k: self._interpolate_env_variables(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_env_variables(elem) for elem in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var_name = data[2:-1]
            return os.getenv(env_var_name, data)
        return data

    def _load_config(self):
        loaded = False
        for path in self._config_paths:
            if os.path.exists(path):
                logger.info(f"Loading configuration from {path}")
                with open(path, "r") as f:
                    raw_config = yaml.safe_load(f)

                interpolated_config = self._interpolate_env_variables(raw_config)

                try:
                    if self._config:
                        file_config_data = interpolated_config.get("llm", {})
                        current_providers = [
                            p.model_dump() for p in self._config.providers
                        ]
                        current_virtual = (
                            [v.model_dump() for v in self._config.virtual]
                            if self._config.virtual
                            else []
                        )

                        new_providers = file_config_data.get("providers", [])
                        all_providers = {p["name"]: p for p in current_providers}
                        for p_data in new_providers:
                            try:
                                p = LLMProviderConfig(**p_data)
                                all_providers[p.name] = p.model_dump()
                            except ValidationError as e:
                                logger.warning(
                                    f"Skipping invalid provider '{p_data.get('name', 'UNKNOWN')}' from config file: {e}"
                                )

                        new_virtual = file_config_data.get("virtual", [])
                        all_virtual = {v["name"]: v for v in current_virtual}
                        for v in new_virtual:
                            all_virtual[v["name"]] = v

                        combined_config_data = {
                            "providers": list(all_providers.values()),
                            "virtual": list(all_virtual.values()),
                            "default_model": file_config_data.get(
                                "default_model", self._config.default_model
                            ),
                        }
                        self._config = LLMConfig(**combined_config_data)
                    else:
                        self._config = LLMConfig(**interpolated_config["llm"])

                    loaded = True
                    break
                except KeyError:
                    logger.warning(f"Configuration file {path} is missing 'llm' key.")
                except ValidationError as e:
                    logger.error(f"Configuration validation error for {path}: {e}")
                except Exception as e:
                    logger.error(f"Error loading configuration from {path}: {e}")

        self._config_loaded = loaded
        if not loaded and not self._config:
            logger.info(
                f"No configuration file found at {', '.join(self._config_paths)}. Proceeding with environment variables and defaults only."
            )
            self._config = LLMConfig(providers=[], virtual=[], default_model=None)

    def signal_429(self, provider_name: str, duration: Optional[int] = None):
        """Signal that a provider received a 429 error and should be put on cooldown.

        Args:
            provider_name: Name of the provider that received the 429 error
            duration: Optional specific cooldown duration in seconds. If not provided,
                     uses the configured cooldown duration for this provider.
        """
        if duration is None:
            duration = self.get_cooldown_duration(provider_name)

        with self._lock:
            self._unusable_providers[provider_name] = time.time() + duration
            logger.info(
                f"Provider '{provider_name}' put on cooldown for {duration} seconds"
            )

    def _is_provider_unusable(self, provider_name: str) -> bool:
        with self._lock:
            if provider_name in self._unusable_providers:
                cooldown_end = self._unusable_providers[provider_name]
                current_time = time.time()
                if current_time < cooldown_end:
                    return True
                else:
                    del self._unusable_providers[provider_name]
            return False

    def get(
        self,
        name: Optional[str] = None,
        approximate_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        allow_await_cooldown: bool = True,
    ) -> LLMProviderConfig:
        # Default provider logic
        if name is None:
            if self._default_provider_name:
                name = self._default_provider_name
                logger.info(f"Using programmatically set default provider: {name}")
            elif self._config and self._config.default_model:
                name = self._config.default_model
                logger.info(f"Using default provider from config file: {name}")
            elif self._real_providers:
                name = next(iter(self._real_providers))
                logger.info(f"Using first available provider: {name}")
            else:
                raise ValueError(
                    "No default LLM provider specified and no configuration file found. "
                    "Please specify a provider name in provider:model format, set a default in borg.yaml, or use set_default_provider()."
                )

        # Use configured timeout if none provided
        if timeout is None:
            timeout = self.get_timeout_duration(name)

        provider_key = None
        model_name_for_request = None

        if ":" in name:
            parts = name.split(":", 1)
            provider_key = parts[0]
            model_name_for_request = parts[1]

        # Check if this is a built-in provider request
        if provider_key and provider_key in BUILTIN_PROVIDERS:
            with _GLOBAL_BUILTIN_LOCK:
                if provider_key in _GLOBAL_BUILTIN_PROVIDERS:
                    provider_instance = _GLOBAL_BUILTIN_PROVIDERS[provider_key]

                    if (
                        model_name_for_request
                        and provider_instance.model != model_name_for_request
                    ):
                        provider_instance.model = model_name_for_request

                    if allow_await_cooldown:
                        self._await_cooldown(provider_key, timeout=timeout)
                    if self._is_provider_unusable(provider_key):
                        raise ValueError(
                            f"Provider '{provider_key}' is on cooldown and await_cooldown is false"
                        )

                    if provider_instance.has_multiple_keys():
                        provider_instance.get_next_api_key()

                    self._real_providers[provider_key] = provider_instance
                    return provider_instance

                settings = BUILTIN_PROVIDERS[provider_key]
                api_key_env = settings["api_key_env"]
                api_keys_env = api_key_env + "S"

                api_keys_value = os.getenv(api_keys_env)
                api_key_value = os.getenv(api_key_env)

                api_keys_list = []
                if api_keys_value:
                    api_keys_list = _parse_api_keys(api_keys_value)
                elif api_key_value:
                    api_keys_list = _parse_api_keys(api_key_value)

                if not api_keys_list:
                    env_var_names = (
                        [api_keys_env, api_key_env]
                        if api_keys_value is None
                        else [api_keys_env]
                    )
                    raise ValueError(
                        f"Built-in provider '{provider_key}' requires "
                        f"one of the environment variables {env_var_names} to be set."
                    )

                if not model_name_for_request:
                    model_name_for_request = settings["default_model"]

                if provider_key in _GLOBAL_BUILTIN_PROVIDERS:
                    return _GLOBAL_BUILTIN_PROVIDERS[provider_key]

                builtin_config = LLMProviderConfig(
                    name=provider_key,
                    base_url=settings["base_url"],
                    model=model_name_for_request,
                    api_key=api_keys_list[0],
                    api_keys=api_keys_list,
                    temperature=settings.get("temperature", 0.7),
                    max_tokens=settings.get("max_tokens", 4096),
                )
                _GLOBAL_BUILTIN_PROVIDERS[provider_key] = builtin_config
                return builtin_config

        # Handle configured providers
        if name in self._real_providers:
            if allow_await_cooldown:
                self._await_cooldown(name, timeout=timeout)
            if self._is_provider_unusable(name):
                raise ValueError(
                    f"Provider '{name}' is on cooldown and await_cooldown is false"
                )

            provider = self._real_providers[name]
            if provider.has_multiple_keys():
                provider.get_next_api_key()
            return provider
        elif name in self._virtual_providers:
            return self._get_from_virtual_provider(
                name, approximate_tokens, timeout, allow_await_cooldown
            )
        else:
            # Check if it's a built-in provider without model specification
            if name in BUILTIN_PROVIDERS:
                raise ValueError(
                    f"Provider '{name}' requires model specification. Use format '{name}:model_name' (e.g., '{name}:{BUILTIN_PROVIDERS[name]['default_model']}')"
                )
            raise ValueError(f"LLM provider '{name}' not found")

    def _await_cooldown(
        self, provider_name: str, interval: float = 1.0, timeout: Optional[float] = None
    ):
        with self._lock:
            if provider_name in self._unusable_providers:
                cooldown_end = self._unusable_providers[provider_name]
                current_time = time.time()
                if current_time < cooldown_end:
                    time_to_wait = cooldown_end - current_time
                    if timeout is not None and time_to_wait > timeout:
                        raise TimeoutError(
                            f"Timeout waiting for provider {provider_name} to exit cooldown"
                        )
                    logger.info(
                        f"Provider '{provider_name}' in cooldown. Waiting "
                        f"{time_to_wait:.2f} seconds..."
                    )
                    time.sleep(time_to_wait)
                    if self._is_provider_unusable(provider_name):
                        raise ValueError(
                            f"Provider '{provider_name}' is still in cooldown after waiting."
                        )
                del self._unusable_providers[provider_name]

    def _get_from_virtual_provider(
        self,
        virtual_provider_name: str,
        approximate_tokens: Optional[int],
        timeout: Optional[float],
        allow_await_cooldown: bool,
    ) -> LLMProviderConfig:
        virtual_config = self._virtual_providers[virtual_provider_name]
        start_time = time.time()

        while True:
            # First clear all expired providers to ensure consistent state
            current_time = time.time()
            expired_providers = [
                provider_name
                for provider_name, cooldown_end in self._unusable_providers.items()
                if cooldown_end <= current_time
            ]
            for provider_name in expired_providers:
                del self._unusable_providers[provider_name]

            all_resolved_upstreams: List[LLMProviderConfig] = []
            min_cooldown_end_time = float("inf")

            for upstream_info in virtual_config.upstreams:
                upstream_name = upstream_info["name"]

                if self._is_provider_unusable(upstream_name):
                    current_cooldown_end = self._unusable_providers.get(
                        upstream_name, float("inf")
                    )
                    min_cooldown_end_time = min(
                        min_cooldown_end_time, current_cooldown_end
                    )
                    continue

                try:
                    resolved_provider = self.get(
                        upstream_name,
                        approximate_tokens,
                        timeout=None,
                        allow_await_cooldown=False,
                    )
                    all_resolved_upstreams.append(resolved_provider)
                except ValueError:
                    pass

            if all_resolved_upstreams:
                filtered_upstreams: List[LLMProviderConfig] = []
                if approximate_tokens is not None:
                    for provider in all_resolved_upstreams:
                        if approximate_tokens <= provider.max_tokens:
                            filtered_upstreams.append(provider)
                else:
                    filtered_upstreams = all_resolved_upstreams

                if filtered_upstreams:
                    import random

                    selected_provider = filtered_upstreams[0]
                    if selected_provider.has_multiple_keys():
                        selected_provider.get_next_api_key()
                    return selected_provider

            if not allow_await_cooldown:
                raise ValueError(
                    f"No eligible upstream providers for virtual provider {virtual_provider_name}. All are on cooldown."
                )

            if min_cooldown_end_time == float("inf"):
                raise ValueError(
                    f"No upstreams found for virtual provider {virtual_provider_name} to await."
                )

            time_to_wait = min_cooldown_end_time - time.time()
            if time_to_wait <= 0:
                # Cooldowns have expired, restart the loop to pick up available providers
                continue

            if timeout is not None:
                time_elapsed = time.time() - start_time
                remaining_timeout = timeout - time_elapsed
                # Use small epsilon to avoid floating point precision issues
                if remaining_timeout <= 1e-6:
                    earliest_provider = None
                    earliest_time = float("inf")
                    for upstream_info in virtual_config.upstreams:
                        upstream_name = upstream_info["name"]
                        if upstream_name in self._unusable_providers:
                            cooldown_end = self._unusable_providers[upstream_name]
                            if cooldown_end < earliest_time:
                                earliest_time = cooldown_end
                                earliest_provider = upstream_name

                    if earliest_provider:
                        raise TimeoutError(
                            f"Timeout waiting for provider {earliest_provider} to exit cooldown."
                        )
                    else:
                        raise ValueError(
                            f"Timeout of {timeout} seconds reached while waiting for usable upstreams for virtual provider {virtual_provider_name}."
                        )
                sleep_duration = min(time_to_wait, remaining_timeout)
                time.sleep(sleep_duration)
            else:
                time.sleep(time_to_wait)
