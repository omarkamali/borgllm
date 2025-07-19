#!/usr/bin/env python3
"""
Test script to verify virtual provider round-robin works correctly
after rate limiting fixes.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from borgllm.borgllm import BorgLLM
from borgllm.langchain import create_llm


# Mocked time class for testing
class MockTime:
    def __init__(self, initial_time=0.0):
        self._time = initial_time

    def time(self):
        return self._time

    def sleep(self, seconds):
        print(f"[MOCKED TIME] Sleeping for {seconds:.2f} seconds...")
        self._time += seconds


# Define test configuration
TEST_CONFIG = {
    "llm": {
        "providers": [
            {
                "name": "provider_a",
                "base_url": "https://api.provider-a.com/v1",
                "model": "model-a",
                "api_key": "key-a",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            {
                "name": "provider_b",
                "base_url": "https://api.provider-b.com/v1",
                "model": "model-b",
                "api_key": "key-b",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        ],
        "virtual": [
            {
                "name": "virtual_test",
                "upstreams": [
                    {"name": "provider_a"},
                    {"name": "provider_b"},
                ],
            }
        ],
    }
}


@pytest.fixture(autouse=True)
def reset_borgllm_singleton():
    """Reset the BorgLLM singleton instance before each test."""
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


def test_virtual_provider_switching():
    """Test that virtual provider deterministically selects the first available upstream."""

    # Mock time for this test - patch both time module and borgllm module
    mock_time = MockTime(initial_time=100.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        # Clear environment variables to prevent built-in providers
        with patch.dict(os.environ, {}, clear=True):
            # Initialize BorgLLM with test config
            client = create_llm("virtual_test", initial_config_data=TEST_CONFIG)

            # 1. Initial state: Should pick provider_a (first in list)
            initial_config = client._get_fresh_config_and_update()
            assert initial_config.name == "provider_a"
            assert str(client.openai_api_base) == "https://api.provider-a.com/v1"
            assert (
                str(client.client._client.base_url) == "https://api.provider-a.com/v1/"
            )

            # 2. Signal 429 for provider_a
            BorgLLM.get_instance().signal_429(
                "provider_a", duration=1
            )  # Short cooldown

            # 3. Get fresh config again: Should now pick provider_b
            after_429_config = client._get_fresh_config_and_update()
            assert after_429_config.name == "provider_b"
            assert str(client.openai_api_base) == "https://api.provider-b.com/v1"
            assert (
                str(client.client._client.base_url) == "https://api.provider-b.com/v1/"
            )

            # 4. Wait for provider_a to be available again
            mock_time.sleep(1.1)  # Wait slightly more than cooldown duration

            # 5. Get fresh config again: Should pick provider_a (now first available again)
            after_cooldown_config = client._get_fresh_config_and_update()
            assert after_cooldown_config.name == "provider_a"
            assert str(client.openai_api_base) == "https://api.provider-a.com/v1"
            assert (
                str(client.client._client.base_url) == "https://api.provider-a.com/v1/"
            )

            # 6. Signal 429 for provider_a and provider_b, and try to get a provider immediately (should fail)
            BorgLLM.get_instance().signal_429(
                "provider_a", duration=100
            )  # Long cooldown for a
            BorgLLM.get_instance().signal_429("provider_b", duration=100)
            with pytest.raises(
                ValueError,
                match="No eligible upstream providers for virtual provider virtual_test. All are on cooldown.",
            ):
                BorgLLM.get_instance().get("virtual_test", allow_await_cooldown=False)

            # 7. Try to get a provider with allow_await_cooldown=True but with a short timeout
            with pytest.raises(
                TimeoutError, match="Timeout waiting for provider .* to exit cooldown."
            ):
                BorgLLM.get_instance().get(
                    "virtual_test", timeout=0.1, allow_await_cooldown=True
                )


def test_virtual_provider_429_switching():
    """Test that virtual provider switches providers when one gets a 429 error and reverts to deterministic order."""

    # Clear environment variables to prevent built-in providers
    with patch.dict(os.environ, {}, clear=True):
        client = create_llm("virtual_test", initial_config_data=TEST_CONFIG)

        # 1. Initial state: Should pick provider_a (first in list)
        initial_config = client._get_fresh_config_and_update()
        assert initial_config.name == "provider_a"

        # 2. Signal 429 for provider_a
        BorgLLM.get_instance().signal_429("provider_a", duration=1)

        # 3. Get a new configuration - should switch to provider_b
        second_config = client._get_fresh_config_and_update()
        assert second_config.name == "provider_b"
        # Verify client config is updated correctly
        assert client.openai_api_base == "https://api.provider-b.com/v1"
        assert client.model_name == "model-b"
        assert client.openai_api_key == "key-b"

        # 4. Wait for provider_a cooldown to expire
        time.sleep(1.1)

        # 5. Get fresh config again: Should pick provider_a again (first available)
        third_config = client._get_fresh_config_and_update()
        assert third_config.name == "provider_a"
        # Verify client config is updated correctly
        assert client.openai_api_base == "https://api.provider-a.com/v1"
        assert client.model_name == "model-a"
        assert client.openai_api_key == "key-a"


def test_nested_virtual_providers():
    """Test nested virtual providers work correctly with deterministic selection."""

    nested_config = {
        "llm": {
            "providers": [
                {
                    "name": "provider_a",
                    "base_url": "https://api.provider-a.com/v1",
                    "model": "model-a",
                    "api_key": "key-a",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "provider_b",
                    "base_url": "https://api.provider-b.com/v1",
                    "model": "model-b",
                    "api_key": "key-b",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "provider_c",
                    "base_url": "https://api.provider-c.com/v1",
                    "model": "model-c",
                    "api_key": "key-c",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            ],
            "virtual": [
                {
                    "name": "virtual_x",
                    "upstreams": [
                        {"name": "provider_a"},
                        {"name": "provider_b"},
                    ],
                },
                {
                    "name": "virtual_y",
                    "upstreams": [
                        {"name": "virtual_x"},
                        {"name": "provider_c"},
                    ],
                },
            ],
        }
    }

    # Clear environment variables to prevent built-in providers
    with patch.dict(os.environ, {}, clear=True):
        client = create_llm("virtual_y", initial_config_data=nested_config)

        # 1. Initial state: virtual_y should pick virtual_x, which in turn picks provider_a
        config1 = client._get_fresh_config_and_update()
        assert config1.name == "provider_a"
        assert str(client.openai_api_base) == "https://api.provider-a.com/v1"

        # 2. Signal 429 for provider_a. virtual_x should now pick provider_b.
        # So virtual_y should now resolve to provider_b.
        BorgLLM.get_instance().signal_429("provider_a", duration=1)
        config2 = client._get_fresh_config_and_update()
        assert config2.name == "provider_b"
        assert str(client.openai_api_base) == "https://api.provider-b.com/v1"

        # 3. Signal 429 for provider_b. Now virtual_x has no available upstreams.
        # So virtual_y should fall back to provider_c (its second upstream).
        BorgLLM.get_instance().signal_429("provider_b", duration=1)
        config3 = client._get_fresh_config_and_update()
        assert config3.name == "provider_c"
        assert str(client.openai_api_base) == "https://api.provider-c.com/v1"

        # 4. Wait for provider_a and provider_b to clear cooldowns
        time.sleep(1.1)

        # 5. After cooldown, virtual_y should again pick virtual_x which picks provider_a
        config4 = client._get_fresh_config_and_update()
        assert config4.name == "provider_a"


def test_nested_virtual_providers_with_429():
    """Test nested virtual providers handle 429 errors correctly with deterministic selection."""

    nested_config = {
        "llm": {
            "providers": [
                {
                    "name": "provider_a",
                    "base_url": "https://api.provider-a.com/v1",
                    "model": "model-a",
                    "api_key": "key-a",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "provider_b",
                    "base_url": "https://api.provider-b.com/v1",
                    "model": "model-b",
                    "api_key": "key-b",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "provider_c",
                    "base_url": "https://api.provider-c.com/v1",
                    "model": "model-c",
                    "api_key": "key-c",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            ],
            "virtual": [
                {
                    "name": "virtual_x",
                    "upstreams": [
                        {"name": "provider_a"},
                        {"name": "provider_b"},
                    ],
                },
                {
                    "name": "virtual_y",
                    "upstreams": [
                        {"name": "virtual_x"},
                        {"name": "provider_c"},
                    ],
                },
            ],
        }
    }

    # Clear environment variables to prevent built-in providers
    with patch.dict(os.environ, {}, clear=True):
        client = create_llm("virtual_y", initial_config_data=nested_config)

        # 1. Initially, virtual_y -> virtual_x -> provider_a
        config1 = client._get_fresh_config_and_update()
        assert config1.name == "provider_a"

        # 2. Signal 429 for provider_a and provider_b (making virtual_x unavailable)
        BorgLLM.get_instance().signal_429("provider_a", duration=1)
        BorgLLM.get_instance().signal_429("provider_b", duration=1)

        # 3. Now virtual_y should only resolve to provider_c
        config2 = client._get_fresh_config_and_update()
        assert config2.name == "provider_c"

        # Verify client config is correct
        assert client.openai_api_base == "https://api.provider-c.com/v1"
        assert client.model_name == "model-c"
        assert client.openai_api_key == "key-c"

        # 4. Wait for cooldowns to expire for provider_a and provider_b
        time.sleep(1.1)

        # 5. After cooldown, virtual_y should again pick virtual_x which picks provider_a
        config3 = client._get_fresh_config_and_update()
        assert config3.name == "provider_a"
        assert client.openai_api_base == "https://api.provider-a.com/v1"
        assert client.model_name == "model-a"
        assert client.openai_api_key == "key-a"


def test_virtual_provider_availability():
    """Test virtual provider availability checking."""
    # Mock time for this test
    mock_time = MockTime(initial_time=200.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        # Clear environment variables to prevent built-in providers
        with patch.dict(os.environ, {}, clear=True):
            borgllm = BorgLLM(initial_config_data=TEST_CONFIG, _force_reinitialize=True)

            # Initially both providers should be available
            assert not borgllm._is_provider_unusable("provider_a")
            assert not borgllm._is_provider_unusable("provider_b")

            # Virtual provider should return provider_a (first in list)
            provider = borgllm.get("virtual_test")
            assert provider.name == "provider_a"

            # Signal 429 for provider_a
            borgllm.signal_429("provider_a", duration=5)

            # provider_a should now be unusable
            assert borgllm._is_provider_unusable("provider_a")
            assert not borgllm._is_provider_unusable("provider_b")

            # Virtual provider should now return provider_b
            provider = borgllm.get("virtual_test")
            assert provider.name == "provider_b"

            # Wait for provider_a cooldown to expire
            mock_time.sleep(5.1)

            # provider_a should be available again
            assert not borgllm._is_provider_unusable("provider_a")

            # Virtual provider should return provider_a again (first in list)
            provider = borgllm.get("virtual_test")
            assert provider.name == "provider_a"


def test_virtual_provider_all_down():
    """Test virtual provider behavior when all upstreams are down."""
    # Mock time for this test
    mock_time = MockTime(initial_time=300.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        # Clear environment variables to prevent built-in providers
        with patch.dict(os.environ, {}, clear=True):
            borgllm = BorgLLM(initial_config_data=TEST_CONFIG, _force_reinitialize=True)

            # Put both providers on cooldown
            borgllm.signal_429("provider_a", duration=10)
            borgllm.signal_429("provider_b", duration=10)

            # Both should be unusable
            assert borgllm._is_provider_unusable("provider_a")
            assert borgllm._is_provider_unusable("provider_b")

            # Virtual provider should raise error when allow_await_cooldown=False
            with pytest.raises(
                ValueError,
                match="No eligible upstream providers for virtual provider virtual_test. All are on cooldown.",
            ):
                borgllm.get("virtual_test", allow_await_cooldown=False)

            # With allow_await_cooldown=True but short timeout, should timeout
            with pytest.raises(
                TimeoutError, match="Timeout waiting for provider .* to exit cooldown."
            ):
                borgllm.get("virtual_test", timeout=1.0, allow_await_cooldown=True)


def test_virtual_provider_partial_recovery():
    """Test virtual provider behavior when some upstreams recover."""
    # Mock time for this test
    mock_time = MockTime(initial_time=400.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        # Clear environment variables to prevent built-in providers
        with patch.dict(os.environ, {}, clear=True):
            borgllm = BorgLLM(initial_config_data=TEST_CONFIG, _force_reinitialize=True)

            # Put both providers on cooldown with different durations
            borgllm.signal_429("provider_a", duration=5)  # Shorter cooldown
            borgllm.signal_429("provider_b", duration=10)  # Longer cooldown

            # Both should be unusable initially
            assert borgllm._is_provider_unusable("provider_a")
            assert borgllm._is_provider_unusable("provider_b")

            # Wait for provider_a to recover
            mock_time.sleep(5.1)

            # provider_a should be available, provider_b still down
            assert not borgllm._is_provider_unusable("provider_a")
            assert borgllm._is_provider_unusable("provider_b")

            # Virtual provider should return provider_a
            provider = borgllm.get("virtual_test")
            assert provider.name == "provider_a"

            # Wait for provider_b to recover
            mock_time.sleep(5)

            # Both should be available
            assert not borgllm._is_provider_unusable("provider_a")
            assert not borgllm._is_provider_unusable("provider_b")

            # Virtual provider should return provider_a (first in list)
            provider = borgllm.get("virtual_test")
            assert provider.name == "provider_a"


def test_virtual_provider_config_update():
    """Test that virtual provider updates when config changes."""
    # Mock time for this test
    mock_time = MockTime(initial_time=500.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        # Clear environment variables to prevent built-in providers
        with patch.dict(os.environ, {}, clear=True):
            # Create LLM client with virtual provider
            llm_client = create_llm("virtual_test", initial_config_data=TEST_CONFIG)

            # Initial config should be provider_a
            initial_config = llm_client._get_fresh_config_and_update()
            assert initial_config.name == "provider_a"

            # Put provider_a on cooldown
            BorgLLM.get_instance().signal_429("provider_a", duration=1)

            # Should switch to provider_b
            new_config = llm_client._get_fresh_config_and_update()
            assert new_config.name == "provider_b"

            # Wait for cooldown to expire
            mock_time.sleep(1.1)

            # Should switch back to provider_a
            recovered_config = llm_client._get_fresh_config_and_update()
            assert recovered_config.name == "provider_a"
