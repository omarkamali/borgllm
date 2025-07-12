"""
Tests for LangChain integration with BorgLLM configuration.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import yaml
from openai import RateLimitError

from borgllm.langchain import (
    BorgLLMLangChainClient,
    create_llm,
)
from borgllm.borgllm import BorgLLM

# Import LangChain components for testing
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


@pytest.fixture(autouse=True)
def reset_borgllm_config_singleton():
    """
    Resets the BorgLLM singleton instance before each test to ensure test isolation.
    """
    # Reset singleton state
    BorgLLM._instance = None
    BorgLLM._config_initialized = False

    # Clear global built-in providers cache
    from borgllm.borgllm import _GLOBAL_BUILTIN_PROVIDERS, _GLOBAL_BUILTIN_LOCK

    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()

    yield

    # Ensure it's clean for subsequent tests if any issue
    BorgLLM._instance = None
    BorgLLM._config_initialized = False

    # Clear global built-in providers cache again
    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()


class TestBorgLLMLangChainClient:
    """Test cases for BorgLLMLangChainClient."""

    def test_client_initialization(self):
        """Test that the client initializes correctly with a given provider."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://api.test.com/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "test-provider")

            assert client.provider_name == "test-provider"
            assert client.model_name == "test-model"
            # openai_api_key is a SecretStr, so we need to get its value
            assert client.openai_api_key.get_secret_value() == "test-key"
            assert client.openai_api_base == "https://api.test.com/v1"
            assert client.temperature == 0.5
            assert client.max_tokens == 1000
        finally:
            os.unlink(config_file)

    def test_config_update_from_provider(self):
        """Test that the client configuration updates when the provider config changes."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "update-test-provider",
                        "base_url": "https://api.update-test.com/v1",
                        "model": "update-test-model",
                        "api_key": "update-test-key",
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    }
                ],
                "default_model": "update-test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "update-test-provider")

            # Get the provider config and modify it
            provider_config = borgllm_config.get("update-test-provider")
            provider_config.model = "new-model"
            provider_config.api_key = "new-key"

            # Update the client configuration
            client._update_config_from_provider(provider_config)

            assert client.model_name == "new-model"
            assert (
                client.openai_api_key == "new-key"
            )  # This should be a string after update
        finally:
            os.unlink(config_file)

    @patch("borgllm.langchain.time.sleep")
    def test_429_handling_with_retry(self, mock_sleep):
        """Test that 429 errors trigger retries with exponential backoff."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "retry-test-provider",
                        "base_url": "https://api.retry-test.com/v1",
                        "model": "retry-test-model",
                        "api_key": "retry-test-key",
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "retry-test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "retry-test-provider")

            # Mock the parent _generate method to raise RateLimitError on first call, succeed on second
            success_result = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Success!"))],
                llm_output={"token_usage": {}, "model_name": "retry-test-model"},
            )

            call_count = 0

            def mock_generate(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError(
                        "Rate limit exceeded",
                        response=Mock(status_code=429),
                        body={"error": {"message": "Rate limit exceeded"}},
                    )
                return success_result

            # Mock signal_429 to prevent actual cooldown
            with patch.object(borgllm_config, "signal_429") as mock_signal_429:
                # Mock the parent class's _generate method instead of the client's
                with patch.object(ChatOpenAI, "_generate", side_effect=mock_generate):
                    from langchain_core.messages import HumanMessage

                    result = client.generate([[HumanMessage(content="Test message")]])

            assert result.generations[0] == success_result.generations
            assert (
                result.llm_output["model_name"]
                == success_result.llm_output["model_name"]
            )
            assert call_count == 2  # Should have retried once
            mock_sleep.assert_called_once()  # Should have slept between retries
            mock_signal_429.assert_called_once_with(
                "retry-test-provider"
            )  # Should have signaled 429
        finally:
            os.unlink(config_file)

    @patch("borgllm.langchain.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        """Test that the client gives up after max retries and raises the exception."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "max-retry-test-provider",
                        "base_url": "https://api.max-retry-test.com/v1",
                        "model": "max-retry-test-model",
                        "api_key": "max-retry-test-key",
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "max-retry-test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "max-retry-test-provider")

            # Mock the parent _generate method to always raise RateLimitError
            def mock_generate(*args, **kwargs):
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )

            # Mock signal_429 to prevent actual cooldown
            with patch.object(borgllm_config, "signal_429") as mock_signal_429:
                # Mock the parent class's _generate method
                with patch.object(ChatOpenAI, "_generate", side_effect=mock_generate):
                    with pytest.raises(RateLimitError):
                        from langchain_core.messages import HumanMessage

                        client.generate([[HumanMessage(content="Test message")]])

            # Should have tried 10 times (max retries) and slept 9 times
            assert mock_sleep.call_count == 9
            # Should have signaled 429 for each attempt
            assert mock_signal_429.call_count == 10
        finally:
            os.unlink(config_file)

    def test_non_rate_limit_errors_propagate(self):
        """Test that non-429 errors are propagated immediately."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://api.test.com/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "test-provider")

            # Mock the parent _generate method to always fail with a non-429 error
            def mock_generate(*args, **kwargs):
                raise ValueError("Some other error")

            # Mock the parent class's _generate method
            with patch.object(ChatOpenAI, "_generate", side_effect=mock_generate):
                with pytest.raises(ValueError, match="Some other error"):
                    from langchain_core.messages import HumanMessage

                    client.generate([[HumanMessage(content="Test message")]])

        finally:
            os.unlink(config_file)

    def test_empty_chat_result_handling(self):
        """Test that an empty ChatResult is handled gracefully without causing an IndexError."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://api.test.com/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "temperature": 0.5,
                        "max_tokens": 1000,
                    }
                ],
                "default_model": "test-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize the singleton with the temporary config file
            borgllm_config = BorgLLM.get_instance(config_path=config_file)
            client = BorgLLMLangChainClient(borgllm_config, "test-provider")

            # Mock the parent _generate method to return a ChatResult with empty generations
            # but still have the expected structure for LangChain
            mock_llm_generate = MagicMock(
                return_value=ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=""))],
                    llm_output={"token_usage": {}, "model_name": "test-model"},
                )
            )

            # Mock the parent class's _generate method
            with patch.object(ChatOpenAI, "_generate", mock_llm_generate):
                # Import HumanMessage for the test
                from langchain_core.messages import HumanMessage

                # Invoking the client should not raise an IndexError
                response = client.invoke([HumanMessage(content="test")])
                assert (
                    response.content == ""
                )  # Should return empty string for empty content

        finally:
            os.unlink(config_file)


class TestConvenienceFunction:
    """Test cases for the create_llm convenience function."""

    def test_create_llm_function(self):
        """Test the create_llm convenience function using the singleton."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "convenience-provider",
                        "base_url": "https://convenience.api/v1",
                        "model": "convenience-model",
                        "api_key": "convenience-key",
                        "max_tokens": 1000,  # Add required field
                    }
                ],
                "default_model": "convenience-provider",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # create_llm will now use BorgLLM.get_instance
            client = create_llm(
                config_file=config_file, provider_name="convenience-provider"
            )

            assert isinstance(client, BorgLLMLangChainClient)
            assert client.provider_name == "convenience-provider"
            assert client.model_name == "convenience-model"

        finally:
            os.unlink(config_file)

    def test_create_llm_function_with_initial_data(self):
        """Test create_llm with initial_config_data for singleton initialization."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "initial-data-provider",
                        "base_url": "https://initial.api/v1",
                        "model": "initial-model",
                        "api_key": "initial-key",
                        "max_tokens": 1000,  # Add required field
                    }
                ],
                "default_model": "initial-data-provider",
            }
        }

        # create_llm will use BorgLLM.get_instance
        # Use a non-existent config file to ensure only dict config is used
        client = create_llm(
            config_file="non_existent_config.yaml",
            initial_config_data=config_data,
            provider_name="initial-data-provider",
        )

        assert isinstance(client, BorgLLMLangChainClient)
        assert client.provider_name == "initial-data-provider"
        assert client.model_name == "initial-model"
