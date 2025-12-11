"""
Tests for OpenAI client integration with BorgLLM.

Tests cover:
- BorgOpenAI and BorgAsyncOpenAI client initialization
- Provider resolution from model IDs
- Rate limit handling and retry logic
- Virtual provider support
- API key rotation
- Streaming support
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import yaml
import asyncio

from openai import RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from borgllm.openai import (
    BorgOpenAI,
    BorgAsyncOpenAI,
    BorgChat,
    BorgChatCompletions,
    BorgAsyncChat,
    BorgAsyncChatCompletions,
    BorgResponses,
    BorgAsyncResponses,
)
from borgllm.borgllm import BorgLLM


@pytest.fixture(autouse=True)
def reset_borgllm_singleton():
    """Reset BorgLLM singleton before each test."""
    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    
    from borgllm.borgllm import _GLOBAL_BUILTIN_PROVIDERS, _GLOBAL_BUILTIN_LOCK
    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()
    
    yield
    
    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    
    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()


def create_mock_chat_completion(content: str = "Hello!") -> ChatCompletion:
    """Create a mock ChatCompletion response."""
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
            )
        ],
        created=1677652288,
        model="gpt-4o",
        object="chat.completion",
    )


class TestBorgOpenAIInitialization:
    """Test BorgOpenAI client initialization."""
    
    def test_basic_initialization(self):
        """Test that BorgOpenAI initializes correctly."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://api.test.com/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "test-provider",
            }
        }
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        assert client._borgllm_config is not None
        assert isinstance(client.chat, BorgChat)
        assert isinstance(client.responses, BorgResponses)
    
    def test_initialization_with_config_file(self):
        """Test initialization with a config file."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "file-provider",
                        "base_url": "https://api.file.com/v1",
                        "model": "file-model",
                        "api_key": "file-key",
                        "max_tokens": 2000,
                    }
                ],
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            client = BorgOpenAI(config_file=config_file)
            assert client._borgllm_config is not None
            assert "file-provider" in client._borgllm_config._real_providers
        finally:
            os.unlink(config_file)
    
    def test_initialization_with_cooldown_config(self):
        """Test initialization with cooldown configuration."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "cooldown-provider",
                        "base_url": "https://api.cooldown.com/v1",
                        "model": "cooldown-model",
                        "api_key": "cooldown-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(
            initial_config_data=config_data,
            cooldown={"cooldown-provider": 120, "default": 60},
        )
        
        assert client._borgllm_config.get_cooldown_duration("cooldown-provider") == 120
        assert client._borgllm_config.get_cooldown_duration("other-provider") == 60
    
    def test_initialization_with_timeout_config(self):
        """Test initialization with timeout configuration."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "timeout-provider",
                        "base_url": "https://api.timeout.com/v1",
                        "model": "timeout-model",
                        "api_key": "timeout-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(
            initial_config_data=config_data,
            timeout={"timeout-provider": 30.0, "default": 60.0},
        )
        
        assert client._borgllm_config.get_timeout_duration("timeout-provider") == 30.0
        assert client._borgllm_config.get_timeout_duration("other-provider") == 60.0


class TestBorgOpenAIProviderResolution:
    """Test provider resolution from model IDs."""
    
    def test_resolve_configured_provider(self):
        """Test resolving a configured provider."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "my-provider",
                        "base_url": "https://api.myprovider.com/v1",
                        "model": "my-model",
                        "api_key": "my-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(initial_config_data=config_data)
        provider_config = client._resolve_provider("my-provider")
        
        assert provider_config.name == "my-provider"
        assert provider_config.model == "my-model"
        assert provider_config.api_key == "my-key"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_resolve_builtin_provider(self):
        """Test resolving a built-in provider."""
        client = BorgOpenAI()
        provider_config = client._resolve_provider("openai:gpt-4o")
        
        assert provider_config.name == "openai"
        assert provider_config.model == "gpt-4o"
        assert provider_config.api_key == "test-openai-key"
    
    def test_resolve_with_overrides(self):
        """Test that overrides are applied during resolution."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "override-provider",
                        "base_url": "https://api.override.com/v1",
                        "model": "original-model",
                        "api_key": "original-key",
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(
            initial_config_data=config_data,
            overrides={"temperature": 0.9},
        )
        provider_config = client._resolve_provider("override-provider")
        
        assert provider_config.temperature == 0.9


class TestBorgOpenAIChatCompletions:
    """Test chat.completions.create() functionality."""
    
    @patch("borgllm.openai.OpenAI")
    def test_chat_completion_success(self, mock_openai_class):
        """Test successful chat completion."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "chat-provider",
                        "base_url": "https://api.chat.com/v1",
                        "model": "chat-model",
                        "api_key": "chat-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Test response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        response = client.chat.completions.create(
            model="chat-provider",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "Test response"
        mock_client.chat.completions.create.assert_called_once()
    
    @patch("borgllm.openai.OpenAI")
    @patch("borgllm.openai.time.sleep")
    def test_chat_completion_rate_limit_retry(self, mock_sleep, mock_openai_class):
        """Test that rate limit errors trigger retry."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "retry-provider",
                        "base_url": "https://api.retry.com/v1",
                        "model": "retry-model",
                        "api_key": "retry-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Success after retry")
        mock_client = MagicMock()
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )
            return mock_completion
        
        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        with patch.object(client._borgllm_config, "signal_429") as mock_signal:
            response = client.chat.completions.create(
                model="retry-provider",
                messages=[{"role": "user", "content": "Hello"}],
            )
        
        assert response.choices[0].message.content == "Success after retry"
        assert call_count == 2
        mock_sleep.assert_called_once()
        mock_signal.assert_called_once_with("retry-provider")
    
    @patch("borgllm.openai.OpenAI")
    @patch("borgllm.openai.time.sleep")
    def test_chat_completion_max_retries_exceeded(self, mock_sleep, mock_openai_class):
        """Test that max retries raises the error."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "max-retry-provider",
                        "base_url": "https://api.maxretry.com/v1",
                        "model": "max-retry-model",
                        "api_key": "max-retry-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data, max_retries=3)
        
        with patch.object(client._borgllm_config, "signal_429"):
            with pytest.raises(RateLimitError):
                client.chat.completions.create(
                    model="max-retry-provider",
                    messages=[{"role": "user", "content": "Hello"}],
                )
        
        assert mock_sleep.call_count == 2  # 3 attempts, 2 sleeps
    
    @patch("borgllm.openai.OpenAI")
    def test_chat_completion_non_rate_limit_error(self, mock_openai_class):
        """Test that non-rate-limit errors propagate immediately."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "error-provider",
                        "base_url": "https://api.error.com/v1",
                        "model": "error-model",
                        "api_key": "error-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("Some other error")
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        with pytest.raises(ValueError, match="Some other error"):
            client.chat.completions.create(
                model="error-provider",
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestBorgOpenAIVirtualProviders:
    """Test virtual provider support."""
    
    @patch("borgllm.openai.OpenAI")
    def test_virtual_provider_resolution(self, mock_openai_class):
        """Test that virtual providers resolve to upstream providers."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "upstream-1",
                        "base_url": "https://api.upstream1.com/v1",
                        "model": "upstream-model-1",
                        "api_key": "upstream-key-1",
                        "max_tokens": 1000,
                    },
                    {
                        "name": "upstream-2",
                        "base_url": "https://api.upstream2.com/v1",
                        "model": "upstream-model-2",
                        "api_key": "upstream-key-2",
                        "max_tokens": 2000,
                    },
                ],
                "virtual": [
                    {
                        "name": "my-virtual",
                        "upstreams": [
                            {"name": "upstream-1"},
                            {"name": "upstream-2"},
                        ],
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Virtual response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        response = client.chat.completions.create(
            model="my-virtual",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "Virtual response"
        # Should have resolved to one of the upstreams
        assert client._current_provider_name in ["upstream-1", "upstream-2"]
    
    @patch("borgllm.openai.OpenAI")
    @patch("borgllm.openai.time.sleep")
    def test_virtual_provider_fallback_on_cooldown(self, mock_sleep, mock_openai_class):
        """Test that virtual providers fall back when upstream is on cooldown."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "primary",
                        "base_url": "https://api.primary.com/v1",
                        "model": "primary-model",
                        "api_key": "primary-key",
                        "max_tokens": 1000,
                    },
                    {
                        "name": "fallback",
                        "base_url": "https://api.fallback.com/v1",
                        "model": "fallback-model",
                        "api_key": "fallback-key",
                        "max_tokens": 1000,
                    },
                ],
                "virtual": [
                    {
                        "name": "failover-virtual",
                        "upstreams": [
                            {"name": "primary"},
                            {"name": "fallback"},
                        ],
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Fallback response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        # Put primary on cooldown
        client._borgllm_config.signal_429("primary", duration=60)
        
        response = client.chat.completions.create(
            model="failover-virtual",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "Fallback response"
        assert client._current_provider_name == "fallback"


class TestBorgOpenAIApiKeyRotation:
    """Test API key rotation functionality."""
    
    @patch("borgllm.openai.OpenAI")
    def test_api_key_rotation(self, mock_openai_class):
        """Test that API keys rotate on successive calls."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "rotation-provider",
                        "base_url": "https://api.rotation.com/v1",
                        "model": "rotation-model",
                        "api_keys": ["key-1", "key-2", "key-3"],
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        # Make multiple calls and track which keys are used
        used_keys = []
        for _ in range(3):
            client.chat.completions.create(
                model="rotation-provider",
                messages=[{"role": "user", "content": "Hello"}],
            )
            used_keys.append(client._current_provider_config.api_key)
        
        # Keys should rotate
        assert len(set(used_keys)) > 1 or len(used_keys) == 1  # At least rotation attempted


class TestBorgChatProxies:
    """Test BorgChat and BorgChatCompletions proxy behavior."""

    def _config(self):
        return {
            "llm": {
                "providers": [
                    {
                        "name": "proxy-provider",
                        "base_url": "https://api.proxy.com/v1",
                        "model": "proxy-model",
                        "api_key": "proxy-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

    def test_chat_property_returns_proxy(self):
        client = BorgOpenAI(initial_config_data=self._config())

        assert isinstance(client.chat, BorgChat)
        assert isinstance(client.chat.completions, BorgChatCompletions)

    @patch.object(BorgOpenAI, "_create_chat_completion")
    def test_chat_completions_delegate_to_internal_method(self, mock_create):
        mock_create.return_value = "delegated"
        client = BorgOpenAI(initial_config_data=self._config())
        messages = [{"role": "user", "content": "hi"}]

        result = client.chat.completions.create(
            model="proxy-provider",
            messages=messages,
        )

        assert result == "delegated"
        mock_create.assert_called_once_with(
            messages=messages,
            model="proxy-provider",
            stream=False,
        )

    @patch.object(BorgOpenAI, "_create_chat_completion")
    def test_chat_completions_support_stream_flag(self, mock_create):
        mock_create.return_value = "stream"
        client = BorgOpenAI(initial_config_data=self._config())
        messages = [{"role": "user", "content": "hi"}]

        client.chat.completions.create(
            model="proxy-provider",
            messages=messages,
            stream=True,
            temperature=0.2,
        )

        mock_create.assert_called_once_with(
            messages=messages,
            model="proxy-provider",
            stream=True,
            temperature=0.2,
        )


class TestBorgAsyncOpenAI:
    """Test BorgAsyncOpenAI client."""
    
    def test_async_initialization(self):
        """Test that BorgAsyncOpenAI initializes correctly."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "async-provider",
                        "base_url": "https://api.async.com/v1",
                        "model": "async-model",
                        "api_key": "async-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgAsyncOpenAI(initial_config_data=config_data)
        
        assert client._borgllm_config is not None
        assert isinstance(client.chat, BorgAsyncChat)
        assert isinstance(client.responses, BorgAsyncResponses)
    
    @pytest.mark.asyncio
    @patch("borgllm.openai.AsyncOpenAI")
    async def test_async_chat_completion(self, mock_async_openai_class):
        """Test async chat completion."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "async-chat-provider",
                        "base_url": "https://api.asyncchat.com/v1",
                        "model": "async-chat-model",
                        "api_key": "async-chat-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Async response")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_async_openai_class.return_value = mock_client
        
        client = BorgAsyncOpenAI(initial_config_data=config_data)
        response = await client.chat.completions.create(
            model="async-chat-provider",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "Async response"
    
    @pytest.mark.asyncio
    @patch("borgllm.openai.AsyncOpenAI")
    @patch("borgllm.openai.asyncio.sleep", new_callable=AsyncMock)
    async def test_async_rate_limit_retry(self, mock_sleep, mock_async_openai_class):
        """Test async rate limit retry."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "async-retry-provider",
                        "base_url": "https://api.asyncretry.com/v1",
                        "model": "async-retry-model",
                        "api_key": "async-retry-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        mock_completion = create_mock_chat_completion("Async success after retry")
        mock_client = MagicMock()
        
        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )
            return mock_completion
        
        mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)
        mock_async_openai_class.return_value = mock_client
        
        client = BorgAsyncOpenAI(initial_config_data=config_data)
        
        with patch.object(client._borgllm_config, "signal_429") as mock_signal:
            response = await client.chat.completions.create(
                model="async-retry-provider",
                messages=[{"role": "user", "content": "Hello"}],
            )
        
        assert response.choices[0].message.content == "Async success after retry"
        assert call_count == 2
        mock_sleep.assert_called_once()
        mock_signal.assert_called_once()


class TestBorgAsyncChatProxies:
    """Test async chat proxy behavior."""

    def _config(self):
        return {
            "llm": {
                "providers": [
                    {
                        "name": "async-proxy",
                        "base_url": "https://api.asyncproxy.com/v1",
                        "model": "async-model",
                        "api_key": "async-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_async_chat_property_returns_proxy(self):
        client = BorgAsyncOpenAI(initial_config_data=self._config())

        assert isinstance(client.chat, BorgAsyncChat)
        assert isinstance(client.chat.completions, BorgAsyncChatCompletions)

    @pytest.mark.asyncio
    async def test_async_chat_completions_delegate(self):
        client = BorgAsyncOpenAI(initial_config_data=self._config())
        messages = [{"role": "user", "content": "hi"}]

        with patch.object(
            BorgAsyncOpenAI, "_create_chat_completion", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = "async-delegated"

            result = await client.chat.completions.create(
                model="async-proxy",
                messages=messages,
            )

            assert result == "async-delegated"
            mock_create.assert_awaited_once_with(
                messages=messages,
                model="async-proxy",
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_async_chat_completions_support_stream_flag(self):
        client = BorgAsyncOpenAI(initial_config_data=self._config())
        messages = [{"role": "user", "content": "hi"}]

        with patch.object(
            BorgAsyncOpenAI, "_create_chat_completion", new_callable=AsyncMock
        ) as mock_create:
            await client.chat.completions.create(
                model="async-proxy",
                messages=messages,
                stream=True,
                temperature=0.1,
            )

            mock_create.assert_awaited_once_with(
                messages=messages,
                model="async-proxy",
                stream=True,
                temperature=0.1,
            )


class TestBorgOpenAIBuiltinProviders:
    """Test built-in provider support."""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch("borgllm.openai.OpenAI")
    def test_openai_builtin_provider(self, mock_openai_class):
        """Test using OpenAI built-in provider."""
        mock_completion = create_mock_chat_completion("OpenAI response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI()
        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "OpenAI response"
        assert client._current_provider_name == "openai"
    
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"})
    @patch("borgllm.openai.OpenAI")
    def test_groq_builtin_provider(self, mock_openai_class):
        """Test using Groq built-in provider."""
        mock_completion = create_mock_chat_completion("Groq response")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        client = BorgOpenAI()
        response = client.chat.completions.create(
            model="groq:llama3-8b-8192",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert response.choices[0].message.content == "Groq response"
        assert client._current_provider_name == "groq"
    
    def test_missing_api_key_error(self):
        """Test that missing API key raises appropriate error."""
        # Ensure no API keys are set
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars that might interfere
            for key in list(os.environ.keys()):
                if "API_KEY" in key:
                    del os.environ[key]
            
            client = BorgOpenAI()
            
            with pytest.raises(ValueError, match="requires"):
                client._resolve_provider("openai:gpt-4o")


class TestDuckTyping:
    """Test that BorgOpenAI duck-types correctly as OpenAI client."""
    
    def test_chat_attribute_exists(self):
        """Test that chat attribute exists."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "duck-provider",
                        "base_url": "https://api.duck.com/v1",
                        "model": "duck-model",
                        "api_key": "duck-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")
    
    def test_responses_attribute_exists(self):
        """Test that responses attribute exists."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "duck-provider",
                        "base_url": "https://api.duck.com/v1",
                        "model": "duck-model",
                        "api_key": "duck-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        client = BorgOpenAI(initial_config_data=config_data)
        
        assert hasattr(client, "responses")
        assert hasattr(client.responses, "create")
