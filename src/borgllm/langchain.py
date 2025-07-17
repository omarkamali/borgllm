"""
LangChain integration for BorgLLM configuration provider.

This module provides automatic LLM client creation with built-in 429 error handling
and configuration updates.
"""

import asyncio
import time
import logging
from typing import Any, Dict, Optional, Union, List
from functools import wraps

from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import BaseModel, ConfigDict

from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage

from .borgllm import BorgLLM, LLMProviderConfig

logger = logging.getLogger(__name__)


class BorgLLMLangChainClient(ChatOpenAI):
    """
    Custom LangChain OpenAI client that integrates with BorgLLM configuration.

    Automatically handles:
    - Configuration updates for each call
    - 429 error detection and notification to BorgLLM
    - Automatic retry with updated configuration
    """

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        borgllm_config: BorgLLM,
        provider_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the LangChain client with BorgLLM integration.

        Args:
            borgllm_config: The BorgLLM configuration instance
            provider_name: Optional name of the provider to use. If None, BorgLLM will use its default logic.
            overrides: Optional dictionary of settings to override.
            **kwargs: Additional arguments passed to ChatOpenAI
        """
        print(
            f"Initializing BorgLLMLangChainClient with provider_name: {provider_name}"
        )
        provider_config = borgllm_config.get(provider_name)

        print(f"Provider config: {provider_config}")
        super().__init__(
            model=provider_config.model,
            api_key=provider_config.api_key,
            base_url=str(provider_config.base_url),
            temperature=provider_config.temperature or 0.7,
            max_tokens=provider_config.max_tokens,
            **kwargs,
        )

        object.__setattr__(self, "_borgllm_config", borgllm_config)
        object.__setattr__(
            self,
            "_provider_name",
            provider_name or borgllm_config._default_provider_name,
        )
        object.__setattr__(
            self, "_current_resolved_provider_name", provider_config.name
        )
        object.__setattr__(self, "_overrides", overrides)

    @property
    def borgllm_config(self) -> BorgLLM:
        """Get the BorgLLM configuration instance."""
        return self._borgllm_config

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    def _update_config_from_provider(self, provider_config: LLMProviderConfig) -> None:
        """Update the client configuration from a provider config."""
        if self._overrides:
            for key, value in self._overrides.items():
                if hasattr(provider_config, key):
                    setattr(provider_config, key, value)

        self.model_name = provider_config.model
        self.openai_api_key = provider_config.api_key
        self.openai_api_base = str(provider_config.base_url)
        self.temperature = provider_config.temperature or 0.7
        self.max_tokens = provider_config.max_tokens

        if hasattr(self, "client") and self.client:
            if hasattr(self.client, "_client") and self.client._client:
                self.client._client.api_key = provider_config.api_key
                self.client._client.base_url = str(provider_config.base_url)
            else:
                self.client.api_key = provider_config.api_key
                self.client.base_url = str(provider_config.base_url)

        if hasattr(self, "async_client") and self.async_client:
            if hasattr(self.async_client, "_client") and self.async_client._client:
                self.async_client._client.api_key = provider_config.api_key
                self.async_client._client.base_url = str(provider_config.base_url)
            else:
                self.async_client.api_key = provider_config.api_key
                self.async_client.base_url = str(provider_config.base_url)

        object.__setattr__(
            self, "_current_resolved_provider_name", provider_config.name
        )

    def _get_fresh_config_and_update(self):
        """Get fresh configuration and update client settings."""
        provider_config = self.borgllm_config.get(
            self.provider_name,
            timeout=30,
            allow_await_cooldown=True,
        )
        self._update_config_from_provider(provider_config)
        return provider_config

    def _handle_rate_limit_error(
        self, e: Exception, retry_count: int, max_retries: int
    ):
        """Handle rate limit errors with proper signaling and retry logic."""
        logger.warning(
            f"Rate limit error for provider {self._current_resolved_provider_name}: {e}"
        )
        self.borgllm_config.signal_429(self._current_resolved_provider_name)

        retry_count += 1
        if retry_count >= max_retries:
            logger.error(
                f"Max retries ({max_retries}) reached for provider {self._current_resolved_provider_name}"
            )
            raise
        return retry_count

    def _handle_non_rate_limit_error(self, e: Exception):
        """Handle non-rate-limit errors with detailed logging."""
        logger.error(
            f"Non-rate-limit error for provider {self._current_resolved_provider_name}: {e}"
        )
        logger.error("--------------------------------")
        logger.error("Config Debug Info:")
        logger.error(f"  base_url: {self.openai_api_base}")
        logger.error(f"  model: {self.model_name}")
        logger.error(f"  temperature: {self.temperature}")
        logger.error(f"  max_tokens: {self.max_tokens}")
        logger.error(f"  provider_name: {self.provider_name}")
        logger.error(f"  resolved_provider: {self._current_resolved_provider_name}")
        logger.error("--------------------------------")
        raise

    def _generate(self, *args, **kwargs):
        """Override _generate to add automatic retry logic and fresh config for each request."""
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                self._get_fresh_config_and_update()
                return super()._generate(*args, **kwargs)

            except RateLimitError as e:
                retry_count = self._handle_rate_limit_error(e, retry_count, max_retries)
                time.sleep(0.1)
                continue

            except Exception as e:
                self._handle_non_rate_limit_error(e)

        raise RuntimeError(f"Failed to complete request after {max_retries} retries")

    async def _agenerate(self, *args, **kwargs):
        """Override _agenerate to add automatic retry logic and fresh config for async calls."""
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                self._get_fresh_config_and_update()
                llm_raw_response = await super()._agenerate(*args, **kwargs)

                if llm_raw_response is None:
                    logger.warning(
                        "Received None response from LangChain's _agenerate for provider %s. Converting to empty ChatResult.",
                        self._current_resolved_provider_name,
                    )
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content=""))],
                        llm_output={},
                    )

                return llm_raw_response

            except RateLimitError as e:
                retry_count = self._handle_rate_limit_error(e, retry_count, max_retries)
                await asyncio.sleep(0.1)
                continue

            except Exception as e:
                if isinstance(
                    e, TypeError
                ) and "'NoneType' object is not iterable" in str(e):
                    logger.warning(
                        "Caught specific NoneType error from LangChain parsing. Returning empty ChatResult to prevent crash."
                    )
                    return ChatResult(
                        generations=[ChatGeneration(message=AIMessage(content=""))],
                        llm_output={},
                    )
                else:
                    self._handle_non_rate_limit_error(e)

        raise RuntimeError(f"Failed to complete request after {max_retries} retries")


def create_llm(
    provider_name: Optional[str] = None,
    config_file: str = "borg.yaml",
    initial_config_data: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    cooldown: Optional[Union[int, Dict[str, int]]] = None,
    timeout: Optional[Union[float, Dict[str, float]]] = None,
    **kwargs,
) -> BorgLLMLangChainClient:
    """
    Convenience function to create a LangChain LLM client directly.

    Args:
        provider_name: Optional name of the provider to use. If None, BorgLLM will use its default logic.
        config_file: Path to the BorgLLM configuration file.
        initial_config_data: Optional initial configuration data as dictionary.
        overrides: Optional dictionary of settings to override.
        cooldown: Optional cooldown configuration. Can be:
            - int: Global cooldown duration in seconds for all providers (default: 60)
            - dict: Provider-specific cooldown durations, e.g., {"openai:gpt-4o": 120, "default": 60}
        timeout: Optional timeout configuration. Can be:
            - float: Global timeout duration in seconds for all operations
            - dict: Provider-specific timeout durations, e.g., {"openai:gpt-4o": 30.0, "default": 60.0}
        **kwargs: Additional arguments passed to the LangChain client

    Returns:
        A LangChain OpenAI client configured with the specified provider
    """
    borgllm_config_instance = BorgLLM.get_instance(
        config_path=config_file, initial_config_data=initial_config_data
    )

    # Set cooldown and timeout configurations if provided
    if cooldown is not None:
        borgllm_config_instance.set_cooldown_config(cooldown)
    if timeout is not None:
        borgllm_config_instance.set_timeout_config(timeout)

    return BorgLLMLangChainClient(
        borgllm_config=borgllm_config_instance,
        provider_name=provider_name,
        overrides=overrides,
        **kwargs,
    )
