#!/usr/bin/env python3
"""
Simple test script to verify multiple API keys functionality.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from borgllm.borgllm import BorgLLM


def test_multiple_api_keys():
    """Test multiple API keys functionality."""
    print("Testing Multiple API Keys Functionality")
    print("=" * 50)

    # Test 1: Config with multiple API keys
    print("\n1. Testing config with api_keys list...")
    config_data = {
        "llm": {
            "providers": [
                {
                    "name": "test-multi-keys",
                    "base_url": "https://api.example.com/v1",
                    "model": "test-model",
                    "api_keys": ["key1", "key2", "key3"],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            ]
        }
    }

    config = BorgLLM(
        config_path="nonexistent.yaml",
        initial_config_data=config_data,
        _force_reinitialize=True,
    )

    print("Round-robin test (should cycle through key1, key2, key3):")
    for i in range(6):
        provider = config.get("test-multi-keys")
        print(f"  Call {i+1}: {provider.api_key}")

    # Test 2: Comma-separated API keys
    print("\n2. Testing comma-separated api_key...")
    config_data2 = {
        "llm": {
            "providers": [
                {
                    "name": "test-comma-keys",
                    "base_url": "https://api.example.com/v1",
                    "model": "test-model",
                    "api_key": "keyA,keyB,keyC",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            ]
        }
    }

    config2 = BorgLLM(
        config_path="nonexistent.yaml",
        initial_config_data=config_data2,
        _force_reinitialize=True,
    )

    print("Round-robin test (should cycle through keyA, keyB, keyC):")
    for i in range(4):
        provider = config2.get("test-comma-keys")
        print(f"  Call {i+1}: {provider.api_key}")

    # Test 3: Precedence test (api_keys over api_key)
    print("\n3. Testing precedence (api_keys should take precedence over api_key)...")
    config_data3 = {
        "llm": {
            "providers": [
                {
                    "name": "test-precedence",
                    "base_url": "https://api.example.com/v1",
                    "model": "test-model",
                    "api_key": "should-be-ignored",
                    "api_keys": ["priority1", "priority2"],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            ]
        }
    }

    config3 = BorgLLM(
        config_path="nonexistent.yaml",
        initial_config_data=config_data3,
        _force_reinitialize=True,
    )

    print("Should use priority1 and priority2 (ignoring 'should-be-ignored'):")
    for i in range(3):
        provider = config3.get("test-precedence")
        print(f"  Call {i+1}: {provider.api_key}")

    # Test 4: Built-in provider environment variables
    print("\n4. Testing built-in provider environment variables...")

    # Set test environment variables
    os.environ["TEST_API_KEYS"] = "env-key1,env-key2,env-key3"
    os.environ["TEST_API_KEY"] = "single-env-key"

    # Mock a built-in provider for testing
    from borgllm.borgllm import BUILTIN_PROVIDERS

    original_providers = BUILTIN_PROVIDERS.copy()

    BUILTIN_PROVIDERS["test-builtin"] = {
        "base_url": "https://api.test.com/v1",
        "api_key_env": "TEST_API_KEY",
        "default_model": "test-model",
        "max_tokens": 4096,
    }

    try:
        config4 = BorgLLM(config_path="nonexistent.yaml", _force_reinitialize=True)

        print("Testing built-in provider with TEST_API_KEYS env var:")
        # Check if the built-in provider was added
        if "test-builtin" in config4._real_providers:
            provider_obj = config4._real_providers["test-builtin"]
            print(f"Built-in provider added with keys: {provider_obj._api_keys}")
            for i in range(4):
                try:
                    provider = config4.get("test-builtin:custom-model")
                    print(f"  Call {i+1}: {provider.api_key}")
                except Exception as e:
                    print(f"  Call {i+1}: Error - {e}")
        else:
            print("test-builtin provider was not added to _real_providers")

        # Try direct built-in provider call
        print("Testing direct built-in provider calls:")
        for i in range(4):
            try:
                provider = config4.get("test-builtin")
                print(f"  Call {i+1}: {provider.api_key}")
            except Exception as e:
                print(f"  Call {i+1}: Error - {e}")

    finally:
        # Restore original BUILTIN_PROVIDERS
        BUILTIN_PROVIDERS.clear()
        BUILTIN_PROVIDERS.update(original_providers)

        # Clean up environment
        if "TEST_API_KEYS" in os.environ:
            del os.environ["TEST_API_KEYS"]
        if "TEST_API_KEY" in os.environ:
            del os.environ["TEST_API_KEY"]

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    test_multiple_api_keys()
