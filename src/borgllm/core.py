"""
Core shared utilities for BorgLLM client integrations.

This module provides shared logic for config management, rate limit handling,
and retry logic that is used by both LangChain and OpenAI client integrations.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

from openai import RateLimitError

from .borgllm import BorgLLM, LLMProviderConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitHandler:
    """
    Shared rate limit handling logic for BorgLLM client integrations.
    
    Provides methods for:
    - Detecting and handling 429 errors
    - Signaling rate limits to BorgLLM
    - Retry logic with configurable max retries
    """
    
    DEFAULT_MAX_RETRIES = 10
    DEFAULT_RETRY_DELAY = 0.1  # seconds
    
    def __init__(
        self,
        borgllm_config: BorgLLM,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        self.borgllm_config = borgllm_config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def handle_rate_limit_error(
        self,
        error: Exception,
        provider_name: str,
        retry_count: int,
    ) -> int:
        """
        Handle a rate limit error by signaling to BorgLLM and incrementing retry count.
        
        Args:
            error: The rate limit exception
            provider_name: Name of the provider that hit the rate limit
            retry_count: Current retry count
            
        Returns:
            Updated retry count
            
        Raises:
            The original error if max retries exceeded
        """
        logger.warning(f"Rate limit error for provider {provider_name}: {error}")
        self.borgllm_config.signal_429(provider_name)
        
        retry_count += 1
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) reached for provider {provider_name}")
            raise error
        
        return retry_count
    
    def log_non_rate_limit_error(
        self,
        error: Exception,
        provider_name: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Log detailed information about a non-rate-limit error.
        
        Args:
            error: The exception that occurred
            provider_name: Name of the provider
            base_url: Optional base URL for debugging
            model: Optional model name for debugging
        """
        logger.error(f"Non-rate-limit error for provider {provider_name}: {error}")
        logger.error("--------------------------------")
        logger.error("Config Debug Info:")
        if base_url:
            logger.error(f"  base_url: {base_url}")
        if model:
            logger.error(f"  model: {model}")
        logger.error(f"  provider_name: {provider_name}")
        logger.error("--------------------------------")


class ConfigResolver:
    """
    Shared configuration resolution logic for BorgLLM client integrations.
    
    Handles:
    - Resolving provider configurations from model IDs
    - Applying overrides
    - Managing current provider state
    """
    
    def __init__(
        self,
        borgllm_config: BorgLLM,
        overrides: Optional[dict] = None,
    ):
        self.borgllm_config = borgllm_config
        self.overrides = overrides or {}
    
    def resolve_provider(
        self,
        model_id: str,
        timeout: Optional[float] = None,
        allow_await_cooldown: bool = True,
    ) -> LLMProviderConfig:
        """
        Resolve a provider configuration from a model ID.
        
        Args:
            model_id: The model identifier (e.g., "openai:gpt-4o")
            timeout: Optional timeout for waiting on cooldown
            allow_await_cooldown: Whether to wait for cooldown to expire
            
        Returns:
            The resolved LLMProviderConfig
        """
        provider_config = self.borgllm_config.get(
            model_id,
            timeout=timeout or 30,
            allow_await_cooldown=allow_await_cooldown,
        )
        
        # Apply overrides
        if self.overrides:
            for key, value in self.overrides.items():
                if hasattr(provider_config, key):
                    setattr(provider_config, key, value)
        
        return provider_config
    
    def get_fresh_config(
        self,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> LLMProviderConfig:
        """
        Get a fresh configuration for a model ID, waiting for cooldown if necessary.
        
        Args:
            model_id: The model identifier
            timeout: Optional timeout for waiting
            
        Returns:
            Fresh LLMProviderConfig
        """
        return self.resolve_provider(
            model_id,
            timeout=timeout,
            allow_await_cooldown=True,
        )


def with_retry_sync(
    func: Callable[..., T],
    rate_limit_handler: RateLimitHandler,
    get_provider_name: Callable[[], str],
    get_debug_info: Optional[Callable[[], dict]] = None,
) -> Callable[..., T]:
    """
    Decorator factory for adding retry logic to synchronous functions.
    
    Args:
        func: The function to wrap
        rate_limit_handler: RateLimitHandler instance
        get_provider_name: Callable that returns the current provider name
        get_debug_info: Optional callable that returns debug info dict
        
    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        retry_count = 0
        
        while retry_count < rate_limit_handler.max_retries:
            try:
                return func(*args, **kwargs)
            
            except RateLimitError as e:
                retry_count = rate_limit_handler.handle_rate_limit_error(
                    e, get_provider_name(), retry_count
                )
                time.sleep(rate_limit_handler.retry_delay)
                continue
            
            except Exception as e:
                debug_info = get_debug_info() if get_debug_info else {}
                rate_limit_handler.log_non_rate_limit_error(
                    e,
                    get_provider_name(),
                    base_url=debug_info.get("base_url"),
                    model=debug_info.get("model"),
                )
                raise
        
        raise RuntimeError(
            f"Failed to complete request after {rate_limit_handler.max_retries} retries"
        )
    
    return wrapper


async def with_retry_async(
    coro_func: Callable[..., T],
    rate_limit_handler: RateLimitHandler,
    get_provider_name: Callable[[], str],
    get_debug_info: Optional[Callable[[], dict]] = None,
    *args,
    **kwargs,
) -> T:
    """
    Execute an async function with retry logic.
    
    Args:
        coro_func: The async function to call
        rate_limit_handler: RateLimitHandler instance
        get_provider_name: Callable that returns the current provider name
        get_debug_info: Optional callable that returns debug info dict
        *args, **kwargs: Arguments to pass to coro_func
        
    Returns:
        Result of the async function
    """
    retry_count = 0
    
    while retry_count < rate_limit_handler.max_retries:
        try:
            return await coro_func(*args, **kwargs)
        
        except RateLimitError as e:
            retry_count = rate_limit_handler.handle_rate_limit_error(
                e, get_provider_name(), retry_count
            )
            await asyncio.sleep(rate_limit_handler.retry_delay)
            continue
        
        except Exception as e:
            debug_info = get_debug_info() if get_debug_info else {}
            rate_limit_handler.log_non_rate_limit_error(
                e,
                get_provider_name(),
                base_url=debug_info.get("base_url"),
                model=debug_info.get("model"),
            )
            raise
    
    raise RuntimeError(
        f"Failed to complete request after {rate_limit_handler.max_retries} retries"
    )
