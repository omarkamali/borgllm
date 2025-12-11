"""
Tests for the core shared utilities module.

Tests cover:
- RateLimitHandler functionality
- ConfigResolver functionality
- Retry decorators
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio

from openai import RateLimitError

from borgllm.core import (
    RateLimitHandler,
    ConfigResolver,
    with_retry_sync,
    with_retry_async,
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


class TestRateLimitHandler:
    """Test RateLimitHandler class."""
    
    def test_initialization(self):
        """Test RateLimitHandler initialization."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        assert handler.borgllm_config is borgllm
        assert handler.max_retries == 10
        assert handler.retry_delay == 0.1
    
    def test_custom_max_retries(self):
        """Test custom max retries configuration."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm, max_retries=5, retry_delay=0.5)
        
        assert handler.max_retries == 5
        assert handler.retry_delay == 0.5
    
    def test_handle_rate_limit_error_increments_count(self):
        """Test that handle_rate_limit_error increments retry count."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        error = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        
        with patch.object(borgllm, "signal_429") as mock_signal:
            new_count = handler.handle_rate_limit_error(error, "test-provider", 0)
        
        assert new_count == 1
        mock_signal.assert_called_once_with("test-provider")
    
    def test_handle_rate_limit_error_raises_on_max_retries(self):
        """Test that handle_rate_limit_error raises when max retries reached."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm, max_retries=3)
        
        error = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        
        with patch.object(borgllm, "signal_429"):
            with pytest.raises(RateLimitError):
                handler.handle_rate_limit_error(error, "test-provider", 2)  # Already at 2, max is 3
    
    def test_log_non_rate_limit_error(self):
        """Test logging of non-rate-limit errors."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        error = ValueError("Some error")
        
        # Should not raise, just log
        handler.log_non_rate_limit_error(
            error,
            "test-provider",
            base_url="https://api.test.com/v1",
            model="test-model",
        )


class TestConfigResolver:
    """Test ConfigResolver class."""
    
    def test_resolve_provider(self):
        """Test basic provider resolution."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "resolver-provider",
                        "base_url": "https://api.resolver.com/v1",
                        "model": "resolver-model",
                        "api_key": "resolver-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        resolver = ConfigResolver(borgllm)
        
        provider_config = resolver.resolve_provider("resolver-provider")
        
        assert provider_config.name == "resolver-provider"
        assert provider_config.model == "resolver-model"
    
    def test_resolve_provider_with_overrides(self):
        """Test provider resolution with overrides."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "override-provider",
                        "base_url": "https://api.override.com/v1",
                        "model": "override-model",
                        "api_key": "override-key",
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        resolver = ConfigResolver(borgllm, overrides={"temperature": 0.9})
        
        provider_config = resolver.resolve_provider("override-provider")
        
        assert provider_config.temperature == 0.9
    
    def test_get_fresh_config(self):
        """Test getting fresh configuration."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "fresh-provider",
                        "base_url": "https://api.fresh.com/v1",
                        "model": "fresh-model",
                        "api_key": "fresh-key",
                        "max_tokens": 1000,
                    }
                ],
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        resolver = ConfigResolver(borgllm)
        
        provider_config = resolver.get_fresh_config("fresh-provider")
        
        assert provider_config.name == "fresh-provider"


class TestWithRetrySync:
    """Test synchronous retry decorator."""
    
    def test_successful_call(self):
        """Test that successful calls return normally."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        def success_func():
            return "success"
        
        wrapped = with_retry_sync(
            success_func,
            handler,
            lambda: "test-provider",
        )
        
        result = wrapped()
        assert result == "success"
    
    @patch("borgllm.core.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep):
        """Test retry on rate limit error."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        call_count = 0
        def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )
            return "success after retry"
        
        wrapped = with_retry_sync(
            retry_func,
            handler,
            lambda: "test-provider",
        )
        
        with patch.object(borgllm, "signal_429"):
            result = wrapped()
        
        assert result == "success after retry"
        assert call_count == 2
        mock_sleep.assert_called_once()
    
    def test_non_rate_limit_error_propagates(self):
        """Test that non-rate-limit errors propagate immediately."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        def error_func():
            raise ValueError("Some error")
        
        wrapped = with_retry_sync(
            error_func,
            handler,
            lambda: "test-provider",
        )
        
        with pytest.raises(ValueError, match="Some error"):
            wrapped()


class TestWithRetryAsync:
    """Test asynchronous retry function."""
    
    @pytest.mark.asyncio
    async def test_successful_async_call(self):
        """Test that successful async calls return normally."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        async def success_func():
            return "async success"
        
        result = await with_retry_async(
            success_func,
            handler,
            lambda: "test-provider",
        )
        
        assert result == "async success"
    
    @pytest.mark.asyncio
    async def test_async_retry_on_rate_limit(self):
        """Test async retry on rate limit error."""
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
            }
        }
        
        borgllm = BorgLLM.get_instance(initial_config_data=config_data)
        handler = RateLimitHandler(borgllm)
        
        call_count = 0
        async def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )
            return "async success after retry"
        
        with patch.object(borgllm, "signal_429"):
            result = await with_retry_async(
                retry_func,
                handler,
                lambda: "test-provider",
            )
        
        assert result == "async success after retry"
        assert call_count == 2
