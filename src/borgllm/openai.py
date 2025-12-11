"""
OpenAI client integration for BorgLLM.

This module provides BorgOpenAI and BorgAsyncOpenAI clients that are drop-in
replacements for openai.OpenAI and openai.AsyncOpenAI, with automatic:
- Provider resolution from model IDs (e.g., "openai:gpt-4o", "google:gemini-2.5-flash")
- Rate limit handling with automatic retry
- API key rotation
- Virtual provider support
- Cooldown management
"""

import asyncio
import time
import logging
from typing import Any, Dict, Iterator, AsyncIterator, Optional, Union, List, Literal, overload

from openai import OpenAI, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from openai._streaming import Stream, AsyncStream

from .borgllm import BorgLLM, LLMProviderConfig
from .core import RateLimitHandler, ConfigResolver

logger = logging.getLogger(__name__)


class BorgChatCompletions:
    """
    Proxy for chat.completions that handles BorgLLM provider resolution.
    
    Duck-types as openai.resources.chat.Completions.
    """
    
    def __init__(self, borg_client: "BorgOpenAI"):
        self._borg_client = borg_client
    
    @overload
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        **kwargs,
    ) -> Stream[ChatCompletionChunk]:
        ...
    
    @overload
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion:
        ...
    
    @overload
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        ...
    
    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        """
        Create a chat completion with automatic provider resolution.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier in format "provider:model" (e.g., "openai:gpt-4o")
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to the OpenAI API
            
        Returns:
            ChatCompletion or Stream[ChatCompletionChunk] if streaming
        """
        return self._borg_client._create_chat_completion(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs,
        )


class BorgAsyncChatCompletions:
    """
    Async proxy for chat.completions that handles BorgLLM provider resolution.
    
    Duck-types as openai.resources.chat.AsyncCompletions.
    """
    
    def __init__(self, borg_client: "BorgAsyncOpenAI"):
        self._borg_client = borg_client
    
    @overload
    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[True],
        **kwargs,
    ) -> AsyncStream[ChatCompletionChunk]:
        ...
    
    @overload
    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion:
        ...
    
    @overload
    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        ...
    
    async def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        """
        Create a chat completion asynchronously with automatic provider resolution.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier in format "provider:model" (e.g., "openai:gpt-4o")
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to the OpenAI API
            
        Returns:
            ChatCompletion or AsyncStream[ChatCompletionChunk] if streaming
        """
        return await self._borg_client._create_chat_completion(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs,
        )


class BorgChat:
    """
    Proxy for chat namespace.
    
    Duck-types as openai.resources.Chat.
    """
    
    def __init__(self, borg_client: "BorgOpenAI"):
        self._borg_client = borg_client
        self._completions: Optional[BorgChatCompletions] = None
    
    @property
    def completions(self) -> BorgChatCompletions:
        if self._completions is None:
            self._completions = BorgChatCompletions(self._borg_client)
        return self._completions


class BorgAsyncChat:
    """
    Async proxy for chat namespace.
    
    Duck-types as openai.resources.AsyncChat.
    """
    
    def __init__(self, borg_client: "BorgAsyncOpenAI"):
        self._borg_client = borg_client
        self._completions: Optional[BorgAsyncChatCompletions] = None
    
    @property
    def completions(self) -> BorgAsyncChatCompletions:
        if self._completions is None:
            self._completions = BorgAsyncChatCompletions(self._borg_client)
        return self._completions


class BorgResponses:
    """
    Proxy for responses API that handles BorgLLM provider resolution.
    
    Duck-types as openai.resources.Responses.
    """
    
    def __init__(self, borg_client: "BorgOpenAI"):
        self._borg_client = borg_client
    
    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: Literal[True],
        **kwargs,
    ) -> Stream[Response]:
        ...
    
    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: Literal[False] = False,
        **kwargs,
    ) -> Response:
        ...
    
    @overload
    def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, Stream[Response]]:
        ...
    
    def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, Stream[Response]]:
        """
        Create a response with automatic provider resolution.
        
        Args:
            model: Model identifier in format "provider:model" (e.g., "openai:gpt-4o")
            input: The input text or messages
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to the OpenAI API
            
        Returns:
            Response or Stream[Response] if streaming
        """
        return self._borg_client._create_response(
            model=model,
            input=input,
            stream=stream,
            **kwargs,
        )


class BorgAsyncResponses:
    """
    Async proxy for responses API that handles BorgLLM provider resolution.
    
    Duck-types as openai.resources.AsyncResponses.
    """
    
    def __init__(self, borg_client: "BorgAsyncOpenAI"):
        self._borg_client = borg_client
    
    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: Literal[True],
        **kwargs,
    ) -> AsyncStream[Response]:
        ...
    
    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: Literal[False] = False,
        **kwargs,
    ) -> Response:
        ...
    
    @overload
    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, AsyncStream[Response]]:
        ...
    
    async def create(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, AsyncStream[Response]]:
        """
        Create a response asynchronously with automatic provider resolution.
        
        Args:
            model: Model identifier in format "provider:model" (e.g., "openai:gpt-4o")
            input: The input text or messages
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to the OpenAI API
            
        Returns:
            Response or AsyncStream[Response] if streaming
        """
        return await self._borg_client._create_response(
            model=model,
            input=input,
            stream=stream,
            **kwargs,
        )


class BorgOpenAI:
    """
    Drop-in replacement for openai.OpenAI with BorgLLM integration.

    Automatically handles:

    - Provider resolution from model IDs (e.g., "openai:gpt-4o")
    - Rate limit detection and retry
    - API key rotation
    - Virtual provider support
    - Cooldown management

    Example::

        from borgllm import BorgOpenAI

        client = BorgOpenAI()
        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    
    def __init__(
        self,
        config_file: str = "borg.yaml",
        initial_config_data: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        cooldown: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[Union[float, Dict[str, float]]] = None,
        max_retries: int = 10,
        **kwargs,
    ):
        """
        Initialize the BorgOpenAI client.
        
        Args:
            config_file: Path to the BorgLLM configuration file
            initial_config_data: Optional initial configuration data as dictionary
            overrides: Optional dictionary of settings to override
            cooldown: Optional cooldown configuration (int or dict)
            timeout: Optional timeout configuration (float or dict)
            max_retries: Maximum number of retries on rate limit errors
            **kwargs: Additional arguments (reserved for future use)
        """
        self._borgllm_config = BorgLLM.get_instance(
            config_path=config_file,
            initial_config_data=initial_config_data,
        )
        
        if cooldown is not None:
            self._borgllm_config.set_cooldown_config(cooldown)
        if timeout is not None:
            self._borgllm_config.set_timeout_config(timeout)
        
        self._overrides = overrides or {}
        self._max_retries = max_retries
        self._retry_delay = 0.1
        
        # Cache for OpenAI clients per provider
        self._clients: Dict[str, OpenAI] = {}
        
        # Current request state
        self._current_provider_name: Optional[str] = None
        self._current_provider_config: Optional[LLMProviderConfig] = None
        
        # Lazy-initialized proxies
        self._chat: Optional[BorgChat] = None
        self._responses: Optional[BorgResponses] = None
    
    @property
    def chat(self) -> BorgChat:
        """Access the chat completions API."""
        if self._chat is None:
            self._chat = BorgChat(self)
        return self._chat
    
    @property
    def responses(self) -> BorgResponses:
        """Access the responses API."""
        if self._responses is None:
            self._responses = BorgResponses(self)
        return self._responses
    
    def _get_or_create_client(self, provider_config: LLMProviderConfig) -> OpenAI:
        """
        Get or create an OpenAI client for the given provider config.
        
        Args:
            provider_config: The resolved provider configuration
            
        Returns:
            Configured OpenAI client
        """
        # Create a new client with current config (API key may have rotated)
        client = OpenAI(
            api_key=provider_config.api_key,
            base_url=str(provider_config.base_url),
        )
        return client
    
    def _resolve_provider(self, model_id: str) -> LLMProviderConfig:
        """
        Resolve provider configuration from a model ID.
        
        Args:
            model_id: Model identifier (e.g., "openai:gpt-4o")
            
        Returns:
            Resolved LLMProviderConfig
        """
        provider_config = self._borgllm_config.get(
            model_id,
            timeout=30,
            allow_await_cooldown=True,
        )
        
        # Apply overrides
        if self._overrides:
            for key, value in self._overrides.items():
                if hasattr(provider_config, key):
                    setattr(provider_config, key, value)
        
        self._current_provider_name = provider_config.name
        self._current_provider_config = provider_config
        
        return provider_config
    
    def _handle_rate_limit(self, error: Exception, retry_count: int) -> int:
        """Handle rate limit error and return updated retry count."""
        provider_name = self._current_provider_name or "unknown"
        logger.warning(f"Rate limit error for provider {provider_name}: {error}")
        self._borgllm_config.signal_429(provider_name)
        
        retry_count += 1
        if retry_count >= self._max_retries:
            logger.error(f"Max retries ({self._max_retries}) reached for provider {provider_name}")
            raise
        
        return retry_count
    
    def _log_error(self, error: Exception) -> None:
        """Log detailed error information."""
        provider_name = self._current_provider_name or "unknown"
        logger.error(f"Non-rate-limit error for provider {provider_name}: {error}")
        logger.error("--------------------------------")
        logger.error("Config Debug Info:")
        if self._current_provider_config:
            logger.error(f"  base_url: {self._current_provider_config.base_url}")
            logger.error(f"  model: {self._current_provider_config.model}")
        logger.error(f"  provider_name: {provider_name}")
        logger.error("--------------------------------")
    
    def _create_chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        """
        Internal method to create a chat completion with retry logic.
        """
        retry_count = 0
        
        while retry_count < self._max_retries:
            try:
                # Resolve provider fresh for each attempt
                provider_config = self._resolve_provider(model)
                client = self._get_or_create_client(provider_config)
                
                # Make the API call
                return client.chat.completions.create(
                    model=provider_config.model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
            
            except RateLimitError as e:
                retry_count = self._handle_rate_limit(e, retry_count)
                time.sleep(self._retry_delay)
                continue
            
            except Exception as e:
                self._log_error(e)
                raise
        
        raise RuntimeError(f"Failed to complete request after {self._max_retries} retries")
    
    def _create_response(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, Stream[Response]]:
        """
        Internal method to create a response with retry logic.
        """
        retry_count = 0
        
        while retry_count < self._max_retries:
            try:
                # Resolve provider fresh for each attempt
                provider_config = self._resolve_provider(model)
                client = self._get_or_create_client(provider_config)
                
                # Make the API call
                return client.responses.create(
                    model=provider_config.model,
                    input=input,
                    stream=stream,
                    **kwargs,
                )
            
            except RateLimitError as e:
                retry_count = self._handle_rate_limit(e, retry_count)
                time.sleep(self._retry_delay)
                continue
            
            except Exception as e:
                self._log_error(e)
                raise
        
        raise RuntimeError(f"Failed to complete request after {self._max_retries} retries")


class BorgAsyncOpenAI:
    """
    Drop-in replacement for openai.AsyncOpenAI with BorgLLM integration.

    Automatically handles:

    - Provider resolution from model IDs (e.g., "openai:gpt-4o")
    - Rate limit detection and retry
    - API key rotation
    - Virtual provider support
    - Cooldown management

    Example::

        from borgllm import BorgAsyncOpenAI

        client = BorgAsyncOpenAI()
        response = await client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    
    def __init__(
        self,
        config_file: str = "borg.yaml",
        initial_config_data: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        cooldown: Optional[Union[int, Dict[str, int]]] = None,
        timeout: Optional[Union[float, Dict[str, float]]] = None,
        max_retries: int = 10,
        **kwargs,
    ):
        """
        Initialize the BorgAsyncOpenAI client.
        
        Args:
            config_file: Path to the BorgLLM configuration file
            initial_config_data: Optional initial configuration data as dictionary
            overrides: Optional dictionary of settings to override
            cooldown: Optional cooldown configuration (int or dict)
            timeout: Optional timeout configuration (float or dict)
            max_retries: Maximum number of retries on rate limit errors
            **kwargs: Additional arguments (reserved for future use)
        """
        self._borgllm_config = BorgLLM.get_instance(
            config_path=config_file,
            initial_config_data=initial_config_data,
        )
        
        if cooldown is not None:
            self._borgllm_config.set_cooldown_config(cooldown)
        if timeout is not None:
            self._borgllm_config.set_timeout_config(timeout)
        
        self._overrides = overrides or {}
        self._max_retries = max_retries
        self._retry_delay = 0.1
        
        # Cache for AsyncOpenAI clients per provider
        self._clients: Dict[str, AsyncOpenAI] = {}
        
        # Current request state
        self._current_provider_name: Optional[str] = None
        self._current_provider_config: Optional[LLMProviderConfig] = None
        
        # Lazy-initialized proxies
        self._chat: Optional[BorgAsyncChat] = None
        self._responses: Optional[BorgAsyncResponses] = None
    
    @property
    def chat(self) -> BorgAsyncChat:
        """Access the chat completions API."""
        if self._chat is None:
            self._chat = BorgAsyncChat(self)
        return self._chat
    
    @property
    def responses(self) -> BorgAsyncResponses:
        """Access the responses API."""
        if self._responses is None:
            self._responses = BorgAsyncResponses(self)
        return self._responses
    
    def _get_or_create_client(self, provider_config: LLMProviderConfig) -> AsyncOpenAI:
        """
        Get or create an AsyncOpenAI client for the given provider config.
        
        Args:
            provider_config: The resolved provider configuration
            
        Returns:
            Configured AsyncOpenAI client
        """
        # Create a new client with current config (API key may have rotated)
        client = AsyncOpenAI(
            api_key=provider_config.api_key,
            base_url=str(provider_config.base_url),
        )
        return client
    
    def _resolve_provider(self, model_id: str) -> LLMProviderConfig:
        """
        Resolve provider configuration from a model ID.
        
        Args:
            model_id: Model identifier (e.g., "openai:gpt-4o")
            
        Returns:
            Resolved LLMProviderConfig
        """
        provider_config = self._borgllm_config.get(
            model_id,
            timeout=30,
            allow_await_cooldown=True,
        )
        
        # Apply overrides
        if self._overrides:
            for key, value in self._overrides.items():
                if hasattr(provider_config, key):
                    setattr(provider_config, key, value)
        
        self._current_provider_name = provider_config.name
        self._current_provider_config = provider_config
        
        return provider_config
    
    def _handle_rate_limit(self, error: Exception, retry_count: int) -> int:
        """Handle rate limit error and return updated retry count."""
        provider_name = self._current_provider_name or "unknown"
        logger.warning(f"Rate limit error for provider {provider_name}: {error}")
        self._borgllm_config.signal_429(provider_name)
        
        retry_count += 1
        if retry_count >= self._max_retries:
            logger.error(f"Max retries ({self._max_retries}) reached for provider {provider_name}")
            raise
        
        return retry_count
    
    def _log_error(self, error: Exception) -> None:
        """Log detailed error information."""
        provider_name = self._current_provider_name or "unknown"
        logger.error(f"Non-rate-limit error for provider {provider_name}: {error}")
        logger.error("--------------------------------")
        logger.error("Config Debug Info:")
        if self._current_provider_config:
            logger.error(f"  base_url: {self._current_provider_config.base_url}")
            logger.error(f"  model: {self._current_provider_config.model}")
        logger.error(f"  provider_name: {provider_name}")
        logger.error("--------------------------------")
    
    async def _create_chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        """
        Internal method to create a chat completion with retry logic.
        """
        retry_count = 0
        
        while retry_count < self._max_retries:
            try:
                # Resolve provider fresh for each attempt
                provider_config = self._resolve_provider(model)
                client = self._get_or_create_client(provider_config)
                
                # Make the API call
                return await client.chat.completions.create(
                    model=provider_config.model,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )
            
            except RateLimitError as e:
                retry_count = self._handle_rate_limit(e, retry_count)
                await asyncio.sleep(self._retry_delay)
                continue
            
            except Exception as e:
                self._log_error(e)
                raise
        
        raise RuntimeError(f"Failed to complete request after {self._max_retries} retries")
    
    async def _create_response(
        self,
        *,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        stream: bool = False,
        **kwargs,
    ) -> Union[Response, AsyncStream[Response]]:
        """
        Internal method to create a response with retry logic.
        """
        retry_count = 0
        
        while retry_count < self._max_retries:
            try:
                # Resolve provider fresh for each attempt
                provider_config = self._resolve_provider(model)
                client = self._get_or_create_client(provider_config)
                
                # Make the API call
                return await client.responses.create(
                    model=provider_config.model,
                    input=input,
                    stream=stream,
                    **kwargs,
                )
            
            except RateLimitError as e:
                retry_count = self._handle_rate_limit(e, retry_count)
                await asyncio.sleep(self._retry_delay)
                continue
            
            except Exception as e:
                self._log_error(e)
                raise
        
        raise RuntimeError(f"Failed to complete request after {self._max_retries} retries")
