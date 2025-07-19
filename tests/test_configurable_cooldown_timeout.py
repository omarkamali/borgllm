import os
import pytest
import yaml
import time
import tempfile
from unittest.mock import patch, MagicMock

from borgllm.borgllm import BorgLLM, LLMProviderConfig
from borgllm.langchain import create_llm


class TestConfigurableCooldown:
    """Test cases for configurable cooldown functionality."""

    def setup_method(self):
        """Reset BorgLLM singleton before each test."""
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

    def test_default_cooldown_duration(self):
        """Test that default cooldown duration is 60 seconds."""
        borgllm = BorgLLM(_force_reinitialize=True)
        assert borgllm.get_cooldown_duration("any_provider") == 60

    def test_set_global_cooldown_config(self):
        """Test setting a global cooldown duration."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set global cooldown to 120 seconds
        borgllm.set_cooldown_config(120)

        assert borgllm.get_cooldown_duration("openai") == 120
        assert borgllm.get_cooldown_duration("anthropic:claude-3") == 120
        assert borgllm.get_cooldown_duration("any_provider") == 120

    def test_set_provider_specific_cooldown_config(self):
        """Test setting provider-specific cooldown durations."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set provider-specific cooldowns
        cooldown_config = {
            "openai": 30,
            "anthropic": 90,
            "openai:gpt-4o": 45,  # More specific override
            "default": 75,
        }
        borgllm.set_cooldown_config(cooldown_config)

        # Test provider key matches
        assert borgllm.get_cooldown_duration("openai") == 30
        assert borgllm.get_cooldown_duration("anthropic") == 90

        # Test specific provider:model matches
        assert borgllm.get_cooldown_duration("openai:gpt-4o") == 45
        assert borgllm.get_cooldown_duration("anthropic:claude-3") == 90

        # Test default fallback
        assert borgllm.get_cooldown_duration("unknown_provider") == 75
        assert borgllm.get_cooldown_duration("groq:llama") == 75

    def test_cooldown_priority_exact_match_over_provider_key(self):
        """Test that exact provider:model matches take priority over provider key matches."""
        borgllm = BorgLLM(_force_reinitialize=True)

        cooldown_config = {"openai": 60, "openai:gpt-4o": 120, "default": 30}
        borgllm.set_cooldown_config(cooldown_config)

        # Exact match should take priority
        assert borgllm.get_cooldown_duration("openai:gpt-4o") == 120
        # Provider key match
        assert borgllm.get_cooldown_duration("openai:gpt-3.5") == 60
        # Default fallback
        assert borgllm.get_cooldown_duration("anthropic:claude") == 30

    def test_signal_429_uses_configured_cooldown(self):
        """Test that signal_429 uses configured cooldown duration when none specified."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set provider-specific cooldown
        borgllm.set_cooldown_config({"openai": 45, "default": 30})

        # Signal 429 without specifying duration
        borgllm.signal_429("openai:gpt-4o")

        # Check that provider is in cooldown with correct duration
        assert borgllm._is_provider_unusable("openai:gpt-4o")

        # The cooldown should be around 45 seconds from now
        expected_cooldown_end = time.time() + 45
        actual_cooldown_end = borgllm._unusable_providers["openai:gpt-4o"]
        assert (
            abs(actual_cooldown_end - expected_cooldown_end) < 1
        )  # Allow 1 second tolerance

    def test_signal_429_explicit_duration_overrides_config(self):
        """Test that explicit duration in signal_429 overrides configured cooldown."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set global cooldown to 60 seconds
        borgllm.set_cooldown_config(60)

        # Signal 429 with explicit duration
        borgllm.signal_429("openai:gpt-4o", duration=120)

        # Check that provider is in cooldown with explicit duration
        assert borgllm._is_provider_unusable("openai:gpt-4o")

        # The cooldown should be around 120 seconds from now
        expected_cooldown_end = time.time() + 120
        actual_cooldown_end = borgllm._unusable_providers["openai:gpt-4o"]
        assert (
            abs(actual_cooldown_end - expected_cooldown_end) < 1
        )  # Allow 1 second tolerance

    def test_create_llm_with_global_cooldown(self):
        """Test that create_llm function sets global cooldown correctly."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with global cooldown
            llm = create_llm(
                config_file=config_file, provider_name="test-provider", cooldown=90
            )

            # Get the BorgLLM instance and check cooldown config
            borgllm = llm.borgllm_config
            assert borgllm.get_cooldown_duration("test-provider") == 90
            assert borgllm.get_cooldown_duration("any_provider") == 90

        finally:
            os.unlink(config_file)

    def test_create_llm_with_provider_specific_cooldown(self):
        """Test that create_llm function sets provider-specific cooldown correctly."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with provider-specific cooldowns
            cooldown_config = {"test-provider": 45, "openai": 120, "default": 75}

            llm = create_llm(
                config_file=config_file,
                provider_name="test-provider",
                cooldown=cooldown_config,
            )

            # Get the BorgLLM instance and check cooldown config
            borgllm = llm.borgllm_config
            assert borgllm.get_cooldown_duration("test-provider") == 45
            assert borgllm.get_cooldown_duration("openai") == 120
            assert borgllm.get_cooldown_duration("unknown") == 75

        finally:
            os.unlink(config_file)


class TestConfigurableTimeout:
    """Test cases for configurable timeout functionality."""

    def setup_method(self):
        """Reset BorgLLM singleton before each test."""
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

    def test_default_timeout_is_none(self):
        """Test that default timeout is None (no timeout)."""
        borgllm = BorgLLM(_force_reinitialize=True)
        assert borgllm.get_timeout_duration("any_provider") is None

    def test_set_global_timeout_config(self):
        """Test setting a global timeout duration."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set global timeout to 30.0 seconds
        borgllm.set_timeout_config(30.0)

        assert borgllm.get_timeout_duration("openai") == 30.0
        assert borgllm.get_timeout_duration("anthropic:claude-3") == 30.0
        assert borgllm.get_timeout_duration("any_provider") == 30.0

    def test_set_provider_specific_timeout_config(self):
        """Test setting provider-specific timeout durations."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set provider-specific timeouts
        timeout_config = {
            "openai": 15.0,
            "anthropic": 45.0,
            "openai:gpt-4o": 20.0,  # More specific override
            "default": 25.0,
        }
        borgllm.set_timeout_config(timeout_config)

        # Test provider key matches
        assert borgllm.get_timeout_duration("openai") == 15.0
        assert borgllm.get_timeout_duration("anthropic") == 45.0

        # Test specific provider:model matches
        assert borgllm.get_timeout_duration("openai:gpt-4o") == 20.0
        assert borgllm.get_timeout_duration("anthropic:claude-3") == 45.0

        # Test default fallback
        assert borgllm.get_timeout_duration("unknown_provider") == 25.0
        assert borgllm.get_timeout_duration("groq:llama") == 25.0

    def test_timeout_priority_exact_match_over_provider_key(self):
        """Test that exact provider:model matches take priority over provider key matches."""
        borgllm = BorgLLM(_force_reinitialize=True)

        timeout_config = {"openai": 30.0, "openai:gpt-4o": 60.0, "default": 15.0}
        borgllm.set_timeout_config(timeout_config)

        # Exact match should take priority
        assert borgllm.get_timeout_duration("openai:gpt-4o") == 60.0
        # Provider key match
        assert borgllm.get_timeout_duration("openai:gpt-3.5") == 30.0
        # Default fallback
        assert borgllm.get_timeout_duration("anthropic:claude") == 15.0

    def test_timeout_config_no_default_returns_none(self):
        """Test that providers not in timeout config return None when no default is set."""
        borgllm = BorgLLM(_force_reinitialize=True)

        timeout_config = {
            "openai": 30.0,
            "anthropic": 45.0,
        }
        borgllm.set_timeout_config(timeout_config)

        # Configured providers should return their timeouts
        assert borgllm.get_timeout_duration("openai") == 30.0
        assert borgllm.get_timeout_duration("anthropic") == 45.0

        # Unconfigured providers should return None
        assert borgllm.get_timeout_duration("unknown_provider") is None
        assert borgllm.get_timeout_duration("groq:llama") is None

    def test_get_method_uses_configured_timeout(self):
        """Test that get method uses configured timeout when none is explicitly provided."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        borgllm = BorgLLM(initial_config_data=config_data, _force_reinitialize=True)

        # Set provider-specific timeout
        borgllm.set_timeout_config({"test-provider": 45.0})

        # Mock the _await_cooldown method to verify it's called with correct timeout
        with patch.object(borgllm, "_await_cooldown") as mock_await:
            provider = borgllm.get("test-provider")

            # Verify _await_cooldown was called with the configured timeout
            mock_await.assert_called_once_with("test-provider", timeout=45.0)

    def test_create_llm_with_global_timeout(self):
        """Test that create_llm function sets global timeout correctly."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with global timeout
            llm = create_llm(
                config_file=config_file, provider_name="test-provider", timeout=45.0
            )

            # Get the BorgLLM instance and check timeout config
            borgllm = llm.borgllm_config
            assert borgllm.get_timeout_duration("test-provider") == 45.0
            assert borgllm.get_timeout_duration("any_provider") == 45.0

        finally:
            os.unlink(config_file)

    def test_create_llm_with_provider_specific_timeout(self):
        """Test that create_llm function sets provider-specific timeout correctly."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with provider-specific timeouts
            timeout_config = {"test-provider": 20.0, "openai": 60.0, "default": 30.0}

            llm = create_llm(
                config_file=config_file,
                provider_name="test-provider",
                timeout=timeout_config,
            )

            # Get the BorgLLM instance and check timeout config
            borgllm = llm.borgllm_config
            assert borgllm.get_timeout_duration("test-provider") == 20.0
            assert borgllm.get_timeout_duration("openai") == 60.0
            assert borgllm.get_timeout_duration("unknown") == 30.0

        finally:
            os.unlink(config_file)


class TestCombinedCooldownTimeout:
    """Test cases for using both cooldown and timeout configurations together."""

    def setup_method(self):
        """Reset BorgLLM singleton before each test."""
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

    def test_create_llm_with_both_cooldown_and_timeout(self):
        """Test that create_llm can set both cooldown and timeout configurations."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with both cooldown and timeout
            cooldown_config = {"test-provider": 90, "default": 60}
            timeout_config = {"test-provider": 30.0, "default": 45.0}

            llm = create_llm(
                config_file=config_file,
                provider_name="test-provider",
                cooldown=cooldown_config,
                timeout=timeout_config,
            )

            # Get the BorgLLM instance and check both configs
            borgllm = llm.borgllm_config

            # Check cooldown config
            assert borgllm.get_cooldown_duration("test-provider") == 90
            assert borgllm.get_cooldown_duration("unknown") == 60

            # Check timeout config
            assert borgllm.get_timeout_duration("test-provider") == 30.0
            assert borgllm.get_timeout_duration("unknown") == 45.0

        finally:
            os.unlink(config_file)

    def test_independent_cooldown_timeout_configs(self):
        """Test that cooldown and timeout configurations work independently."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set different provider preferences for cooldown and timeout
        cooldown_config = {"openai": 120, "anthropic": 60, "default": 90}

        timeout_config = {"openai": 15.0, "groq": 30.0, "default": 25.0}

        borgllm.set_cooldown_config(cooldown_config)
        borgllm.set_timeout_config(timeout_config)

        # Test that each config works independently
        assert borgllm.get_cooldown_duration("openai") == 120
        assert borgllm.get_timeout_duration("openai") == 15.0

        assert borgllm.get_cooldown_duration("anthropic") == 60
        assert (
            borgllm.get_timeout_duration("anthropic") == 25.0
        )  # Falls back to default

        assert borgllm.get_cooldown_duration("groq") == 90  # Falls back to default
        assert borgllm.get_timeout_duration("groq") == 30.0


class TestEdgeCases:
    """Test edge cases and error conditions for cooldown and timeout configurations."""

    def setup_method(self):
        """Reset BorgLLM singleton before each test."""
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

    def test_negative_cooldown_values(self):
        """Test that negative cooldown values are handled appropriately."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set negative cooldown
        borgllm.set_cooldown_config(-30)

        # Should still return the negative value (implementation choice)
        assert borgllm.get_cooldown_duration("any_provider") == -30

    def test_zero_cooldown_values(self):
        """Test that zero cooldown values work correctly."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set zero cooldown
        borgllm.set_cooldown_config(0)

        # Should return zero
        assert borgllm.get_cooldown_duration("any_provider") == 0

        # Signal 429 with zero cooldown
        borgllm.signal_429("test_provider")

        # Provider should not be unusable with zero cooldown
        assert not borgllm._is_provider_unusable("test_provider")

    def test_empty_cooldown_config_dict(self):
        """Test that empty cooldown config dict falls back to default."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set empty dict
        borgllm.set_cooldown_config({})

        # Should fall back to global default
        assert borgllm.get_cooldown_duration("any_provider") == 60

    def test_empty_timeout_config_dict(self):
        """Test that empty timeout config dict returns None."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set empty dict
        borgllm.set_timeout_config({})

        # Should return None for any provider
        assert borgllm.get_timeout_duration("any_provider") is None

    def test_negative_timeout_values(self):
        """Test that negative timeout values are handled appropriately."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set negative timeout
        borgllm.set_timeout_config(-10.0)

        # Should still return the negative value (implementation choice)
        assert borgllm.get_timeout_duration("any_provider") == -10.0

    def test_zero_timeout_values(self):
        """Test that zero timeout values work correctly."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Set zero timeout
        borgllm.set_timeout_config(0.0)

        # Should return zero
        assert borgllm.get_timeout_duration("any_provider") == 0.0

    def test_provider_name_with_multiple_colons(self):
        """Test provider names with multiple colons in the name."""
        borgllm = BorgLLM(_force_reinitialize=True)

        cooldown_config = {"custom": 45, "custom:model:v1": 90, "default": 30}
        borgllm.set_cooldown_config(cooldown_config)

        # Exact match should work
        assert borgllm.get_cooldown_duration("custom:model:v1") == 90

        # Provider key match should work (splits on first colon)
        assert borgllm.get_cooldown_duration("custom:different:model") == 45

        # Unknown provider should use default
        assert borgllm.get_cooldown_duration("unknown:model:v1") == 30

    def test_very_long_provider_names(self):
        """Test that very long provider names are handled correctly."""
        borgllm = BorgLLM(_force_reinitialize=True)

        long_provider_name = "a" * 1000 + ":model"
        long_provider_key = "a" * 1000

        cooldown_config = {long_provider_key: 120, "default": 60}
        borgllm.set_cooldown_config(cooldown_config)

        # Should match the long provider key
        assert borgllm.get_cooldown_duration(long_provider_name) == 120

    def test_unicode_provider_names(self):
        """Test that unicode characters in provider names are handled correctly."""
        borgllm = BorgLLM(_force_reinitialize=True)

        unicode_provider = "测试:模型"
        unicode_key = "测试"

        cooldown_config = {unicode_key: 75, "default": 60}
        borgllm.set_cooldown_config(cooldown_config)

        # Should match the unicode provider key
        assert borgllm.get_cooldown_duration(unicode_provider) == 75

    def test_mixed_type_configs(self):
        """Test that mixed int/float values in configs work correctly."""
        borgllm = BorgLLM(_force_reinitialize=True)

        # Mix int and float in timeout config
        timeout_config = {
            "provider1": 30,  # int
            "provider2": 45.5,  # float
            "default": 60.0,  # float
        }
        borgllm.set_timeout_config(timeout_config)

        # Both should be returned as floats
        assert borgllm.get_timeout_duration("provider1") == 30.0
        assert borgllm.get_timeout_duration("provider2") == 45.5
        assert borgllm.get_timeout_duration("unknown") == 60.0

    def test_concurrent_config_updates(self):
        """Test that concurrent config updates work correctly with threading."""
        import threading

        borgllm = BorgLLM(_force_reinitialize=True)

        # Use a counter instead of time.sleep to avoid CI hanging
        counter = [0]

        def update_cooldown():
            for i in range(10):
                borgllm.set_cooldown_config(i * 10)
                counter[0] += 1

        def update_timeout():
            for i in range(10):
                borgllm.set_timeout_config(i * 5.0)
                counter[0] += 1

        def signal_429s():
            for i in range(10):
                borgllm.signal_429(f"provider_{i}")
                counter[0] += 1

        # Start concurrent threads
        threads = [
            threading.Thread(target=update_cooldown),
            threading.Thread(target=update_timeout),
            threading.Thread(target=signal_429s),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=5.0)  # Add timeout to prevent hanging

        # Ensure all threads completed
        assert counter[0] == 30, f"Expected 30 operations, got {counter[0]}"

        # Test should complete without exceptions
        # Final values should be the last ones set
        assert borgllm.get_cooldown_duration("any_provider") == 90
        assert borgllm.get_timeout_duration("any_provider") == 45.0

    def test_create_llm_with_invalid_config_types(self):
        """Test that create_llm handles invalid config types appropriately."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "test-provider",
                        "base_url": "https://test.api/v1",
                        "model": "test-model",
                        "api_key": "test-key",
                        "max_tokens": 1000,
                    }
                ]
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Test with string cooldown (should be treated as is)
            llm = create_llm(
                config_file=config_file,
                provider_name="test-provider",
                cooldown="invalid",  # This will be stored as-is
            )

            borgllm = llm.borgllm_config
            # The invalid config should be stored (implementation choice)
            assert borgllm._cooldown_config == "invalid"

        finally:
            os.unlink(config_file)
