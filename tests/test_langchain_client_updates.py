import os
import pytest
from unittest.mock import patch, MagicMock
from borgllm.langchain import create_llm
from borgllm.borgllm import BorgLLM


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


def test_debug_url_switching():
    """Debug test to understand what's happening with URL switching."""

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

    with patch.dict(os.environ, {}, clear=True):
        client = create_llm("virtual_test", initial_config_data=test_config)

        print("=== Initial State ===")
        print(f"client.openai_api_base: {client.openai_api_base}")
        print(f"client.model_name: {client.model_name}")
        print(f"client.openai_api_key: {client.openai_api_key}")
        print(
            f"client._current_resolved_provider_name: {client._current_resolved_provider_name}"
        )

        # Check the actual OpenAI client structure more thoroughly
        print("\n=== Client Structure Investigation ===")
        client_attrs = [attr for attr in dir(client) if not attr.startswith("_")]
        print(f"Client attributes: {client_attrs}")

        # Check for OpenAI client specifically
        if hasattr(client, "client"):
            print(f"client.client: {client.client}")
            print(f"client.client type: {type(client.client)}")
            if hasattr(client.client, "_client"):
                print(f"client.client._client: {client.client._client}")
                print(f"client.client._client type: {type(client.client._client)}")
                if hasattr(client.client._client, "base_url"):
                    print(
                        f"client.client._client.base_url: {client.client._client.base_url}"
                    )
                if hasattr(client.client._client, "api_key"):
                    print(
                        f"client.client._client.api_key: {client.client._client.api_key}"
                    )

        if hasattr(client, "async_client"):
            print(f"client.async_client: {client.async_client}")
            print(f"client.async_client type: {type(client.async_client)}")
            if hasattr(client.async_client, "_client"):
                print(f"client.async_client._client: {client.async_client._client}")
                print(
                    f"client.async_client._client type: {type(client.async_client._client)}"
                )
                if hasattr(client.async_client._client, "base_url"):
                    print(
                        f"client.async_client._client.base_url: {client.async_client._client.base_url}"
                    )
                if hasattr(client.async_client._client, "api_key"):
                    print(
                        f"client.async_client._client.api_key: {client.async_client._client.api_key}"
                    )

        # Now force a configuration update
        print("\n=== After Fresh Config Update ===")
        fresh_config = client._get_fresh_config_and_update()
        print(f"fresh_config.name: {fresh_config.name}")
        print(f"fresh_config.base_url: {fresh_config.base_url}")

        print(f"client.openai_api_base: {client.openai_api_base}")
        print(f"client.model_name: {client.model_name}")
        print(f"client.openai_api_key: {client.openai_api_key}")
        print(
            f"client._current_resolved_provider_name: {client._current_resolved_provider_name}"
        )

        if hasattr(client, "client") and client.client:
            if hasattr(client.client, "_client"):
                print(
                    f"client.client._client.base_url: {client.client._client.base_url}"
                )
                print(f"client.client._client.api_key: {client.client._client.api_key}")
            else:
                print(
                    f"client.client.base_url: {getattr(client.client, 'base_url', 'NOT FOUND')}"
                )
                print(
                    f"client.client.api_key: {getattr(client.client, 'api_key', 'NOT FOUND')}"
                )

        if hasattr(client, "async_client") and client.async_client:
            if hasattr(client.async_client, "_client"):
                print(
                    f"client.async_client._client.base_url: {client.async_client._client.base_url}"
                )
                print(
                    f"client.async_client._client.api_key: {client.async_client._client.api_key}"
                )
            else:
                print(
                    f"client.async_client.base_url: {getattr(client.async_client, 'base_url', 'NOT FOUND')}"
                )
                print(
                    f"client.async_client.api_key: {getattr(client.async_client, 'api_key', 'NOT FOUND')}"
                )

        # Signal 429 for current provider
        current_provider = client._current_resolved_provider_name
        print(f"\n=== Signaling 429 for {current_provider} ===")
        BorgLLM.get_instance().signal_429(current_provider, duration=1)

        # Get fresh config again - should switch to other provider
        print("\n=== After 429 Signal ===")
        fresh_config_after_429 = client._get_fresh_config_and_update()
        print(f"fresh_config_after_429.name: {fresh_config_after_429.name}")
        print(f"fresh_config_after_429.base_url: {fresh_config_after_429.base_url}")

        print(f"client.openai_api_base: {client.openai_api_base}")
        print(f"client.model_name: {client.model_name}")
        print(f"client.openai_api_key: {client.openai_api_key}")
        print(
            f"client._current_resolved_provider_name: {client._current_resolved_provider_name}"
        )

        if hasattr(client, "client") and client.client:
            if hasattr(client.client, "_client"):
                print(
                    f"client.client._client.base_url: {client.client._client.base_url}"
                )
                print(f"client.client._client.api_key: {client.client._client.api_key}")
            else:
                print(
                    f"client.client.base_url: {getattr(client.client, 'base_url', 'NOT FOUND')}"
                )
                print(
                    f"client.client.api_key: {getattr(client.client, 'api_key', 'NOT FOUND')}"
                )

        if hasattr(client, "async_client") and client.async_client:
            if hasattr(client.async_client, "_client"):
                print(
                    f"client.async_client._client.base_url: {client.async_client._client.base_url}"
                )
                print(
                    f"client.async_client._client.api_key: {client.async_client._client.api_key}"
                )
            else:
                print(
                    f"client.async_client.base_url: {getattr(client.async_client, 'base_url', 'NOT FOUND')}"
                )
                print(
                    f"client.async_client.api_key: {getattr(client.async_client, 'api_key', 'NOT FOUND')}"
                )

        # Verify the provider actually switched
        assert (
            current_provider != fresh_config_after_429.name
        ), "Provider should have switched after 429"
        assert str(fresh_config_after_429.base_url) != str(
            fresh_config.base_url
        ), "Base URL should have changed"
