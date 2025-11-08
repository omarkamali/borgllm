#!/usr/bin/env python3
"""
Test script to verify built-in provider API key rotation.
This test STAYS in the codebase to prevent regression.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from borgllm import BorgLLM, set_default_provider

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_builtin_provider_rotation():
    """Test that built-in providers properly rotate API keys."""
    logger.info("Testing Built-in Provider API Key Rotation")
    logger.info("=" * 60)

    # Set up test environment variables
    original_gemini_keys = os.environ.get("GOOGLE_API_KEYS")
    original_gemini_key = os.environ.get("GOOGLE_API_KEY")

    try:
        # Set test API keys
        os.environ["GOOGLE_API_KEYS"] = "gemini-key-1,gemini-key-2,gemini-key-3"
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ[
                "GOOGLE_API_KEY"
            ]  # Remove single key to test multi-key precedence

        # Test 1: Direct provider calls
        logger.info("\n1. Testing direct built-in provider calls:")
        config = BorgLLM(config_path="nonexistent.yaml")

        for i in range(6):
            provider = config.get("google")
            logger.info(f"  Call {i+1}: google -> {provider.api_key}")

        # Test 2: Provider:model syntax calls
        logger.info("\n2. Testing provider:model syntax calls:")
        config2 = BorgLLM(config_path="nonexistent.yaml")

        for i in range(6):
            provider = config2.get("google:gemini-2.5-flash")
            logger.info(f"  Call {i+1}: google:gemini-2.5-flash -> {provider.api_key}")

        # Test 3: Set as default provider and use implicit calls
        logger.info("\n3. Testing default provider calls:")
        config3 = BorgLLM(config_path="nonexistent.yaml")
        set_default_provider("google:gemini-2.5-flash")

        for i in range(6):
            provider = config3.get()  # No arguments - should use default
            logger.info(f"  Call {i+1}: get() -> {provider.api_key}")

        # Test 4: Check that provider instances are cached
        logger.info("\n4. Testing provider instance caching:")
        config4 = BorgLLM(config_path="nonexistent.yaml")

        provider1 = config4.get("google")
        provider2 = config4.get("google:gemini-2.5-flash")
        provider3 = config4.get("google:gemini-1.5-pro")

        logger.info(f"  google provider id: {id(provider1)}")
        logger.info(f"  google:gemini-2.5-flash provider id: {id(provider2)}")
        logger.info(f"  google:gemini-1.5-pro provider id: {id(provider3)}")
        logger.info(
            f"  Same instance? {id(provider1) == id(provider2) == id(provider3)}"
        )

        # Test 5: Check provider state
        logger.info("\n5. Testing provider internal state:")
        config5 = BorgLLM(config_path="nonexistent.yaml")
        provider = config5.get("google")
        logger.info(f"  Provider name: {provider.name}")
        logger.info(f"  Provider has multiple keys: {provider.has_multiple_keys()}")
        logger.info(f"  Provider _api_keys: {provider._api_keys}")
        logger.info(f"  Provider _current_key_index: {provider._current_key_index}")

        # Make a few calls and check state changes
        for i in range(3):
            current_key = provider.api_key
            logger.info(
                f"  Before call {i+1}: api_key={current_key}, index={provider._current_key_index}"
            )
            config5.get("google")  # This should advance the round-robin
            logger.info(
                f"  After call {i+1}: api_key={provider.api_key}, index={provider._current_key_index}"
            )

    finally:
        # Restore original environment
        if original_gemini_keys is not None:
            os.environ["GOOGLE_API_KEYS"] = original_gemini_keys
        elif "GOOGLE_API_KEYS" in os.environ:
            del os.environ["GOOGLE_API_KEYS"]

        if original_gemini_key is not None:
            os.environ["GOOGLE_API_KEY"] = original_gemini_key

    logger.info("\n" + "=" * 60)
    logger.info("Test completed!")


def test_langchain_integration():
    """Test LangChain integration with built-in provider rotation."""
    logger.info("\n\nTesting LangChain Integration with Built-in Provider Rotation")
    logger.info("=" * 60)

    # Set up test environment variables
    original_openai_keys = os.environ.get("OPENAI_API_KEYS")

    try:
        # Set test API keys
        os.environ["OPENAI_API_KEYS"] = "sk-test1,sk-test2,sk-test3"

        from borgllm.langchain import create_llm

        # Test LangChain client rotation
        logger.info("\n1. Testing LangChain client with built-in provider:")
        client = create_llm("openai:gpt-4o", config_file="nonexistent.yaml")

        logger.info(f"  Initial API key: {client.openai_api_key}")

        # Simulate multiple requests
        for i in range(6):
            # Get fresh config like the LangChain client does
            provider_config = client.borgllm_config.get(client.provider_name)
            client._update_config_from_provider(provider_config)
            logger.info(f"  Request {i+1}: {client.openai_api_key}")

    finally:
        # Restore original environment
        if original_openai_keys is not None:
            os.environ["OPENAI_API_KEYS"] = original_openai_keys
        elif "OPENAI_API_KEYS" in os.environ:
            del os.environ["OPENAI_API_KEYS"]


if __name__ == "__main__":
    test_builtin_provider_rotation()
    test_langchain_integration()
