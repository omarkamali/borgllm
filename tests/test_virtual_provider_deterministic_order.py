import os
import pytest
from unittest.mock import patch, MagicMock
from borgllm.langchain import create_llm
from borgllm.borgllm import BorgLLM
import time


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


class MockTime:
    """Mock the time module for deterministic testing."""

    def __init__(self, initial_time: float):
        self.current_time = initial_time

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float):
        self.current_time += seconds


def test_virtual_provider_deterministic_order():
    """Test that virtual provider deterministically selects the first available upstream."""

    test_config = {
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
            client = create_llm("virtual_test", initial_config_data=test_config)

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

            # 4. Signal 429 for provider_b
            BorgLLM.get_instance().signal_429(
                "provider_b", duration=1
            )  # Short cooldown

            # 5. Wait for provider_a to be available again
            mock_time.sleep(1.1)  # Wait slightly more than cooldown duration

            # 6. Get fresh config again: Should pick provider_a (now first available again)
            after_cooldown_config = client._get_fresh_config_and_update()
            assert after_cooldown_config.name == "provider_a"
            assert str(client.openai_api_base) == "https://api.provider-a.com/v1"
            assert (
                str(client.client._client.base_url) == "https://api.provider-a.com/v1/"
            )

            # 7. Signal 429 for provider_a and provider_b, and try to get a provider immediately (should fail)
            BorgLLM.get_instance().signal_429(
                "provider_a", duration=100
            )  # Long cooldown for a
            BorgLLM.get_instance().signal_429("provider_b", duration=100)
            with pytest.raises(
                ValueError,
                match="No eligible upstream providers for virtual provider virtual_test. All are on cooldown.",
            ):
                BorgLLM.get_instance().get("virtual_test", allow_await_cooldown=False)

            # 8. Try to get a provider with allow_await_cooldown=True but with a short timeout
            with pytest.raises(
                TimeoutError, match="Timeout waiting for provider .* to exit cooldown."
            ):
                BorgLLM.get_instance().get(
                    "virtual_test", timeout=0.1, allow_await_cooldown=True
                )
