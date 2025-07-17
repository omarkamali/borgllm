import os
import pytest
import yaml
import time
import tempfile
from unittest.mock import patch, MagicMock

from borgllm.borgllm import BorgLLM, LLMProviderConfig
from borgllm.langchain import create_llm


class TestCooldownTimeoutIntegration:
    """Integration tests for cooldown and timeout functionality in realistic scenarios."""

    def setup_method(self):
        """Reset BorgLLM singleton before each test."""
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

    def test_real_world_cooldown_behavior(self):
        """Test cooldown behavior in a realistic scenario with multiple providers."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "fast-provider",
                        "base_url": "https://fast.api/v1",
                        "model": "fast-model",
                        "api_key": "fast-key",
                        "max_tokens": 4000,
                    },
                    {
                        "name": "slow-provider",
                        "base_url": "https://slow.api/v1",
                        "model": "slow-model",
                        "api_key": "slow-key",
                        "max_tokens": 8000,
                    },
                ],
                "virtual": [
                    {
                        "name": "fallback-provider",
                        "upstreams": [
                            {"name": "fast-provider"},
                            {"name": "slow-provider"},
                        ],
                    }
                ],
            }
        }

        borgllm = BorgLLM(initial_config_data=config_data, _force_reinitialize=True)

        # Set provider-specific cooldowns
        cooldown_config = {
            "fast-provider": 5,  # Fast recovery for fast provider
            "slow-provider": 30,  # Longer recovery for slow provider
            "default": 15,
        }
        borgllm.set_cooldown_config(cooldown_config)

        # Initially, fast-provider should be available
        fast_provider = borgllm.get("fast-provider")
        assert fast_provider.name == "fast-provider"

        # Signal 429 on fast-provider
        borgllm.signal_429("fast-provider")

        # Fast provider should now be in cooldown
        assert borgllm._is_provider_unusable("fast-provider")

        # Virtual provider should fall back to slow-provider
        fallback_provider = borgllm.get("fallback-provider")
        assert fallback_provider.name == "slow-provider"

        # Signal 429 on slow-provider too
        borgllm.signal_429("slow-provider")

        # Both providers should be in cooldown
        assert borgllm._is_provider_unusable("fast-provider")
        assert borgllm._is_provider_unusable("slow-provider")

        # Wait for fast-provider cooldown to expire (5 seconds + tolerance)
        # For testing, we'll simulate by manipulating the cooldown time
        cooldown_end_time = borgllm._unusable_providers["fast-provider"]
        borgllm._unusable_providers["fast-provider"] = time.time() - 1  # Set to past

        # Fast provider should now be available again
        assert not borgllm._is_provider_unusable("fast-provider")

        # But slow provider should still be in cooldown
        assert borgllm._is_provider_unusable("slow-provider")

        # Virtual provider should use fast-provider again
        fallback_provider = borgllm.get("fallback-provider")
        assert fallback_provider.name == "fast-provider"

    def test_timeout_configuration_with_virtual_providers(self):
        """Test timeout configuration working with virtual providers."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "timeout-provider",
                        "base_url": "https://timeout.api/v1",
                        "model": "timeout-model",
                        "api_key": "timeout-key",
                        "max_tokens": 4000,
                    }
                ],
                "virtual": [
                    {
                        "name": "timeout-virtual",
                        "upstreams": [{"name": "timeout-provider"}],
                    }
                ],
            }
        }

        borgllm = BorgLLM(initial_config_data=config_data, _force_reinitialize=True)

        # Set provider-specific timeouts
        timeout_config = {
            "timeout-provider": 10.0,
            "timeout-virtual": 15.0,
            "default": 30.0,
        }
        borgllm.set_timeout_config(timeout_config)

        # Test that get method passes correct timeout to _await_cooldown
        with patch.object(borgllm, "_await_cooldown") as mock_await:
            provider = borgllm.get("timeout-provider")
            mock_await.assert_called_once_with("timeout-provider", timeout=10.0)

        # Test virtual provider timeout
        borgllm.signal_429("timeout-provider")  # Put provider in cooldown

        with patch.object(borgllm, "_await_cooldown") as mock_await:
            with patch.object(borgllm, "_get_from_virtual_provider") as mock_virtual:
                mock_virtual.return_value = borgllm._real_providers["timeout-provider"]

                provider = borgllm.get("timeout-virtual")

                # The virtual provider should be called with the configured timeout
                mock_virtual.assert_called_once_with(
                    "timeout-virtual", None, 15.0, True
                )

    def test_create_llm_integration_with_real_usage_pattern(self):
        """Test create_llm with cooldown and timeout in a realistic usage pattern."""
        # Clean up any existing instances
        BorgLLM._instance = None
        BorgLLM._config_initialized = False

        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "production-provider",
                        "base_url": "https://prod.api/v1",
                        "model": "production-model",
                        "api_key": "prod-key",
                        "max_tokens": 4000,
                    },
                    {
                        "name": "backup-provider",
                        "base_url": "https://backup.api/v1",
                        "model": "backup-model",
                        "api_key": "backup-key",
                        "max_tokens": 2000,
                    },
                ],
                "virtual": [
                    {
                        "name": "auto-fallback",
                        "upstreams": [
                            {"name": "production-provider"},
                            {"name": "backup-provider"},
                        ],
                    }
                ],
                "default_model": "auto-fallback",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Create LLM with production-grade cooldown and timeout settings
            cooldown_config = {
                "production-provider": 120,  # 2 minutes for production
                "backup-provider": 60,  # 1 minute for backup
                "default": 90,
            }

            timeout_config = {
                "production-provider": 30.0,  # 30 seconds for production
                "backup-provider": 45.0,  # 45 seconds for backup (slower)
                "auto-fallback": 60.0,  # 1 minute for virtual provider
                "default": 25.0,
            }

            # Create the LLM client
            llm = create_llm(
                config_file=config_file,
                cooldown=cooldown_config,
                timeout=timeout_config,
            )

            borgllm = llm.borgllm_config

            # Verify configurations are applied correctly
            assert borgllm.get_cooldown_duration("production-provider") == 120
            assert borgllm.get_cooldown_duration("backup-provider") == 60
            assert borgllm.get_timeout_duration("production-provider") == 30.0
            assert borgllm.get_timeout_duration("backup-provider") == 45.0
            assert borgllm.get_timeout_duration("auto-fallback") == 60.0

            # Test the default model usage
            with patch.object(borgllm, "_get_from_virtual_provider") as mock_virtual:
                mock_virtual.return_value = borgllm._real_providers[
                    "production-provider"
                ]

                # Getting with no name should use default (auto-fallback)
                provider = borgllm.get()

                # Should call virtual provider with correct timeout
                mock_virtual.assert_called_once_with("auto-fallback", None, 60.0, True)

            # Test 429 handling with configured cooldowns
            borgllm.signal_429("production-provider")

            # Should be in cooldown
            assert borgllm._is_provider_unusable("production-provider")

            # Cooldown end time should be about 120 seconds from now
            expected_end = time.time() + 120
            actual_end = borgllm._unusable_providers["production-provider"]
            assert abs(actual_end - expected_end) < 2  # Allow 2 second tolerance

        finally:
            os.unlink(config_file)

    def test_dynamic_config_updates_during_operation(self):
        """Test that cooldown and timeout configs can be updated dynamically during operation."""
        config_data = {
            "llm": {
                "providers": [
                    {
                        "name": "dynamic-provider",
                        "base_url": "https://dynamic.api/v1",
                        "model": "dynamic-model",
                        "api_key": "dynamic-key",
                        "max_tokens": 4000,
                    }
                ]
            }
        }

        borgllm = BorgLLM(initial_config_data=config_data, _force_reinitialize=True)

        # Initial configuration
        borgllm.set_cooldown_config(60)
        borgllm.set_timeout_config(30.0)

        assert borgllm.get_cooldown_duration("dynamic-provider") == 60
        assert borgllm.get_timeout_duration("dynamic-provider") == 30.0

        # Signal 429 with initial config
        borgllm.signal_429("dynamic-provider")
        initial_cooldown_end = borgllm._unusable_providers["dynamic-provider"]

        # Update configuration dynamically
        new_cooldown_config = {"dynamic-provider": 15, "default": 45}
        new_timeout_config = {"dynamic-provider": 10.0, "default": 20.0}

        borgllm.set_cooldown_config(new_cooldown_config)
        borgllm.set_timeout_config(new_timeout_config)

        # Verify new configurations are active
        assert borgllm.get_cooldown_duration("dynamic-provider") == 15
        assert borgllm.get_timeout_duration("dynamic-provider") == 10.0
        assert borgllm.get_cooldown_duration("unknown-provider") == 45
        assert borgllm.get_timeout_duration("unknown-provider") == 20.0

        # New 429 signals should use new configuration
        borgllm.signal_429(
            "dynamic-provider"
        )  # This will override the previous cooldown
        new_cooldown_end = borgllm._unusable_providers["dynamic-provider"]

        # New cooldown should be shorter (15 seconds vs 60 seconds)
        expected_new_end = time.time() + 15
        assert abs(new_cooldown_end - expected_new_end) < 1

    def test_stress_test_multiple_providers_and_configs(self):
        """Stress test with many providers and frequent config changes."""
        # Create config with many providers
        providers = []
        for i in range(10):
            providers.append(
                {
                    "name": f"provider-{i}",
                    "base_url": f"https://api{i}.test/v1",
                    "model": f"model-{i}",
                    "api_key": f"key-{i}",
                    "max_tokens": 1000 + i * 500,
                }
            )

        config_data = {"llm": {"providers": providers}}
        borgllm = BorgLLM(initial_config_data=config_data, _force_reinitialize=True)

        # Set complex configurations
        cooldown_config = {f"provider-{i}": 10 + i * 5 for i in range(10)}
        cooldown_config["default"] = 60

        timeout_config = {f"provider-{i}": 5.0 + i * 2.5 for i in range(10)}
        timeout_config["default"] = 30.0

        borgllm.set_cooldown_config(cooldown_config)
        borgllm.set_timeout_config(timeout_config)

        # Test all providers have correct configurations
        for i in range(10):
            provider_name = f"provider-{i}"
            expected_cooldown = 10 + i * 5
            expected_timeout = 5.0 + i * 2.5

            assert borgllm.get_cooldown_duration(provider_name) == expected_cooldown
            assert borgllm.get_timeout_duration(provider_name) == expected_timeout

        # Signal 429 on multiple providers rapidly
        for i in range(0, 10, 2):  # Signal on even-numbered providers
            borgllm.signal_429(f"provider-{i}")

        # Verify they're all in cooldown
        for i in range(0, 10, 2):
            assert borgllm._is_provider_unusable(f"provider-{i}")

        # Verify odd-numbered providers are still available
        for i in range(1, 10, 2):
            assert not borgllm._is_provider_unusable(f"provider-{i}")

        # Test that each provider can still be retrieved (if not in cooldown)
        for i in range(1, 10, 2):  # Test odd-numbered (available) providers
            provider = borgllm.get(f"provider-{i}")
            assert provider.name == f"provider-{i}"
