#!/usr/bin/env python3
"""
Test script that mimics the user's exact scenario.
This test STAYS in the codebase to prevent regression.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from borgllm import BorgLLM, set_default_provider

load_dotenv()

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_user_scenario():
    """Test the exact scenario the user reported."""
    # Reset singleton state when running directly (not through pytest)
    from borgllm.borgllm import BorgLLM, _GLOBAL_BUILTIN_PROVIDERS, _GLOBAL_BUILTIN_LOCK

    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()

    logger.info("Testing User Scenario - Built-in Provider Round Robin")
    logger.info("=" * 60)

    # Set up test environment variables like user's scenario
    original_gemini_keys = os.environ.get("GOOGLE_API_KEYS")

    try:
        # Set test API keys - simulate user having multiple real keys
        os.environ["GOOGLE_API_KEYS"] = "key1-test,key2-test,key3-test"

        # User's setup
        set_default_provider("google:gemini-2.5-flash")

        # Test: Create BorgLLM once, then make multiple calls
        logger.info("\n1. User scenario - one config, multiple calls:")
        borgllm = BorgLLM(config_path="nonexistent.yaml")

        for i in range(6):
            logger.info(f"Run {i+1}")
            provider = borgllm.get()  # Uses default provider
            logger.info(f"  API Key: {provider.api_key}")
            logger.info(f"  Model: {provider.model}")
            logger.info("-" * 40)

        # Test: What happens if we create NEW config instances?
        logger.info("\n2. What if we create NEW config instances each time:")
        for i in range(6):
            logger.info(f"Run {i+1}")
            borgllm_new = BorgLLM(
                config_path="nonexistent.yaml"
            )  # NEW instance each time
            provider = borgllm_new.get()
            logger.info(f"  API Key: {provider.api_key}")
            logger.info(f"  Model: {provider.model}")
            logger.info("-" * 40)

        # Test: Mixed calls with explicit provider names
        logger.info("\n3. Mixed explicit calls:")
        borgllm3 = BorgLLM(config_path="nonexistent.yaml")

        calls = [
            "google:gemini-2.5-flash",
            None,  # Default
            "google",
            "google:gemini-1.5-pro",
            None,  # Default
            "google:gemini-2.5-flash",
        ]

        for i, call in enumerate(calls):
            logger.info(f"Run {i+1}: get({call})")
            if call is None:
                provider = borgllm3.get()
            else:
                provider = borgllm3.get(call)
            logger.info(f"  API Key: {provider.api_key}")
            logger.info(f"  Model: {provider.model}")
            logger.info("-" * 40)

    finally:
        # Restore original environment
        if original_gemini_keys is not None:
            os.environ["GOOGLE_API_KEYS"] = original_gemini_keys
        elif "GOOGLE_API_KEYS" in os.environ:
            del os.environ["GOOGLE_API_KEYS"]

    logger.info("\n" + "=" * 60)
    logger.info("Test completed!")


async def test_with_basedrone_simulation():
    """Simulate what might be happening with BaseDrone."""
    logger.info("\n\nTesting BaseDrone-like Scenario")
    logger.info("=" * 60)

    # Set up test environment variables
    original_gemini_keys = os.environ.get("GOOGLE_API_KEYS")

    try:
        os.environ["GOOGLE_API_KEYS"] = "sim-key1,sim-key2,sim-key3"
        set_default_provider("google:gemini-2.5-flash")

        # Simulate what the user might be doing
        borgllm = BorgLLM(config_path="nonexistent.yaml")  # Created once outside loop

        for i in range(4):
            logger.info(f"Run {i}")

            # Simulate creating a drone/client that might create its own config
            class MockDrone:
                def __init__(self):
                    # This is probably what's happening - new config each time
                    self.borgllm_config = BorgLLM(config_path="nonexistent.yaml")

                async def request(self, message):
                    provider = self.borgllm_config.get()  # Uses default
                    return (
                        f"Response using {provider.api_key} with model {provider.model}"
                    )

            drone = MockDrone()
            result = await drone.request("test message")
            logger.info(f"  Result: {result}")
            logger.info("-" * 50)

    finally:
        # Restore original environment
        if original_gemini_keys is not None:
            os.environ["GOOGLE_API_KEYS"] = original_gemini_keys
        elif "GOOGLE_API_KEYS" in os.environ:
            del os.environ["GOOGLE_API_KEYS"]
