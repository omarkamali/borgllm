#!/usr/bin/env python3
"""
Example demonstrating multiple API keys and round-robin functionality in BorgLLM.

This example shows:
1. How to configure multiple API keys for providers
2. Built-in provider support for *_API_KEYS environment variables
3. Round-robin behavior when making multiple calls
4. Priority rules for api_keys vs api_key
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
from borgllm.borgllm import BorgLLM

# Load environment variables from .env file
load_dotenv()


def setup_demo_environment():
    """
    Set up demo environment variables for testing.
    In real usage, you would set these in your .env file or system environment.
    """
    # Example: Multiple OpenAI API keys
    os.environ["OPENAI_API_KEYS"] = "sk-demo-key1,sk-demo-key2,sk-demo-key3"

    # Example: Multiple Gemini API keys (built-in provider)
    os.environ["GOOGLE_API_KEYS"] = "gemini-key-1,gemini-key-2"

    # Single keys (for comparison)
    os.environ["OPENAI_API_KEY"] = "sk-single-key"
    os.environ["GOOGLE_API_KEY"] = "gemini-single-key"


def demonstrate_config_based_multiple_keys():
    """Demonstrate multiple API keys configured in YAML/dict config."""
    print("\n=== CONFIG-BASED MULTIPLE API KEYS ===")

    # Configuration with multiple API keys
    config_data = {
        "llm": {
            "providers": [
                {
                    "name": "multi-key-gpt",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_keys": ["key1", "key2", "key3"],  # List format
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "comma-separated-gpt",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "key1,key2,key3",  # Comma-separated format
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "precedence-test",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o",
                    "api_key": "single-key",  # This will be ignored
                    "api_keys": ["multi-key-1", "multi-key-2"],  # This takes precedence
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            ]
        }
    }

    config = BorgLLM(config_path="nonexistent.yaml", initial_config_data=config_data)

    # Test round-robin behavior
    print("Testing round-robin with multi-key-gpt:")
    for i in range(5):
        provider = config.get("multi-key-gpt")
        print(f"  Call {i+1}: Using API key '{provider.api_key}'")

    print("\nTesting comma-separated format:")
    for i in range(3):
        provider = config.get("comma-separated-gpt")
        print(f"  Call {i+1}: Using API key '{provider.api_key}'")

    print("\nTesting api_keys precedence over api_key:")
    for i in range(3):
        provider = config.get("precedence-test")
        print(
            f"  Call {i+1}: Using API key '{provider.api_key}' (should be multi-key-*)"
        )


def demonstrate_builtin_provider_multiple_keys():
    """Demonstrate multiple API keys with built-in providers."""
    print("\n=== BUILT-IN PROVIDER MULTIPLE API KEYS ===")

    # Pass a nonexistent config_path to ensure only built-in env vars are used
    config = BorgLLM(config_path="nonexistent.yaml")

    # Test OpenAI built-in provider with multiple keys
    if "OPENAI_API_KEYS" in os.environ:
        print("Testing OpenAI built-in provider with OPENAI_API_KEYS:")
        for i in range(4):
            try:
                provider = config.get("openai:gpt-4o")
                print(f"  Call {i+1}: Using API key '{provider.api_key[:20]}...'")
            except ValueError as e:
                print(f"  Call {i+1}: Error - {e}")

    # Test Gemini built-in provider with multiple keys
    if "GOOGLE_API_KEYS" in os.environ:
        print("\nTesting Google (Gemini) built-in provider with GOOGLE_API_KEYS:")
        for i in range(3):
            try:
                provider = config.get("google:gemini-2.5-flash")
                print(f"  Call {i+1}: Using API key '{provider.api_key[:20]}...'")
            except ValueError as e:
                print(f"  Call {i+1}: Error - {e}")


def demonstrate_priority_rules():
    """Demonstrate priority rules: *_API_KEYS takes precedence over *_API_KEY."""
    print("\n=== PRIORITY RULES DEMONSTRATION ===")

    # Test with both OPENAI_API_KEY and OPENAI_API_KEYS set
    original_single = os.environ.get("OPENAI_API_KEY")
    original_multi = os.environ.get("OPENAI_API_KEYS")

    try:
        # Set both single and multiple keys
        os.environ["OPENAI_API_KEY"] = "single-key-should-be-ignored"
        os.environ["OPENAI_API_KEYS"] = "multi-key-1,multi-key-2,multi-key-3"

        config = BorgLLM(config_path="nonexistent.yaml")

        print("Both OPENAI_API_KEY and OPENAI_API_KEYS are set:")
        print(f"  OPENAI_API_KEY = '{os.environ['OPENAI_API_KEY']}'")
        print(f"  OPENAI_API_KEYS = '{os.environ['OPENAI_API_KEYS']}'")

        print(
            "\nTesting which keys are actually used (should be from OPENAI_API_KEYS):"
        )
        for i in range(4):
            provider = config.get("openai:gpt-4o")
            print(f"  Call {i+1}: Using API key '{provider.api_key}'")

    finally:
        # Restore original environment
        if original_single is not None:
            os.environ["OPENAI_API_KEY"] = original_single
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if original_multi is not None:
            os.environ["OPENAI_API_KEYS"] = original_multi
        elif "OPENAI_API_KEYS" in os.environ:
            del os.environ["OPENAI_API_KEYS"]


def demonstrate_virtual_provider_roundrobin():
    """Demonstrate round-robin behavior with virtual providers."""
    print("\n=== VIRTUAL PROVIDER ROUND-ROBIN ===")

    config_data = {
        "llm": {
            "providers": [
                {
                    "name": "provider-a",
                    "base_url": "https://api.example.com/v1",
                    "model": "model-a",
                    "api_keys": ["provider-a-key1", "provider-a-key2"],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
                {
                    "name": "provider-b",
                    "base_url": "https://api.example.com/v1",
                    "model": "model-b",
                    "api_keys": [
                        "provider-b-key1",
                        "provider-b-key2",
                        "provider-b-key3",
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            ],
            "virtual": [
                {
                    "name": "multi-provider",
                    "upstreams": [{"name": "provider-a"}, {"name": "provider-b"}],
                }
            ],
        }
    }

    config = BorgLLM(config_path="nonexistent.yaml", initial_config_data=config_data)

    print(
        "Testing virtual provider with round-robin on both provider selection AND API keys:"
    )
    for i in range(6):
        provider = config.get("multi-provider")
        print(
            f"  Call {i+1}: Provider '{provider.name}' using API key '{provider.api_key}'"
        )


def main():
    """Main demo function."""
    print("BorgLLM Multiple API Keys Demo")
    print("=" * 50)

    # Set up demo environment
    setup_demo_environment()

    try:
        # Run demonstrations
        demonstrate_config_based_multiple_keys()
        demonstrate_builtin_provider_multiple_keys()
        demonstrate_priority_rules()
        demonstrate_virtual_provider_roundrobin()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Multiple API keys via 'api_keys' field (list or comma-separated)")
        print("✓ Comma-separated values in 'api_key' field")
        print("✓ Built-in provider support for *_API_KEYS environment variables")
        print("✓ Priority rules: api_keys > api_key, *_API_KEYS > *_API_KEY")
        print("✓ Round-robin behavior for both provider selection and API keys")
        print("✓ Backward compatibility with single API keys")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
