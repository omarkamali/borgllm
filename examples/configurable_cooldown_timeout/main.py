#!/usr/bin/env python3
"""
Example demonstrating configurable cooldown and timeout functionality in BorgLLM.

This example shows how to configure and observe custom cooldown periods
for providers and set request timeouts using `create_llm`.

It uses a `borg.yaml` file in this directory to define providers and can
override those settings programmatically.

Run this with:
    python main.py
"""

import os
import time
from dotenv import load_dotenv
from borgllm import create_llm, BorgLLM
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Path to the borg.yaml for this example
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "borg.yaml")


def main():
    """Run examples to demonstrate configurable cooldown and timeout functionality."""
    print("BorgLLM Configurable Cooldown and Timeout Example")
    print("=" * 50)

    # Check if we have required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("\nNote: OPENAI_API_KEY not found in environment.")
        print("This example will use a dummy provider for demonstration, ")
        print("but actual API calls would require valid API keys.")

    try:
        # --- Example 1: Global Cooldown and Timeout ---
        print("\n=== Example 1: Global Cooldown (10s) and Timeout (5.0s) ===")
        BorgLLM._instance = None  # Reset for fresh configuration
        BorgLLM._config_initialized = False

        # Create LLM with global cooldown of 10 seconds and timeout of 5.0 seconds
        llm_global = create_llm(
            config_file=CONFIG_FILE,
            provider_name="test-provider-1",  # Defined in borg.yaml
            cooldown=10,  # 10 seconds cooldown for all providers
            timeout=5.0,  # 5.0 seconds timeout for all operations
        )
        borgllm_global = llm_global.borgllm_config

        print(
            f"  Configured Cooldown for 'test-provider-1': {borgllm_global.get_cooldown_duration('test-provider-1')}s"
        )
        print(
            f"  Configured Timeout for 'test-provider-1': {borgllm_global.get_timeout_duration('test-provider-1')}s"
        )
        print(
            f"  Configured Cooldown for 'any_other_provider': {borgllm_global.get_cooldown_duration('any_other_provider')}s"
        )
        print(
            f"  Configured Timeout for 'any_other_provider': {borgllm_global.get_timeout_duration('any_other_provider')}s"
        )

        # Simulate a 429 error and observe cooldown
        print(
            "  Simulating 429 error on 'test-provider-1' (should be on cooldown for 10s)... "
        )
        borgllm_global.signal_429("test-provider-1")
        print(
            f"  Is 'test-provider-1' unusable? {borgllm_global._is_provider_unusable('test-provider-1')}"
        )
        # In a real scenario, the next call would wait or fallback based on allow_await_cooldown
        # time.sleep(10) # Uncomment to actually wait

        # --- Example 2: Provider-Specific Cooldown and Timeout ---
        print("\n=== Example 2: Provider-Specific Cooldown and Timeout ===")
        BorgLLM._instance = None  # Reset for fresh configuration
        BorgLLM._config_initialized = False

        cooldown_config = {
            "test-provider-2": 5,  # Shorter cooldown for this specific provider
            "openai": 60,  # Example for a built-in provider
            "default": 15,  # Default for others
        }
        timeout_config = {
            "test-provider-2": 2.0,  # Shorter timeout
            "openai": 10.0,  # Example for a built-in provider
            "default": 7.0,  # Default for others
        }

        llm_specific = create_llm(
            config_file=CONFIG_FILE,
            provider_name="test-provider-2",  # Defined in borg.yaml
            cooldown=cooldown_config,
            timeout=timeout_config,
        )
        borgllm_specific = llm_specific.borgllm_config

        print(
            f"  Configured Cooldown for 'test-provider-2': {borgllm_specific.get_cooldown_duration('test-provider-2')}s"
        )
        print(
            f"  Configured Timeout for 'test-provider-2': {borgllm_specific.get_timeout_duration('test-provider-2')}s"
        )
        print(
            f"  Configured Cooldown for 'openai:gpt-4o': {borgllm_specific.get_cooldown_duration('openai:gpt-4o')}s"
        )
        print(
            f"  Configured Timeout for 'openai:gpt-4o': {borgllm_specific.get_timeout_duration('openai:gpt-4o')}s"
        )
        print(
            f"  Configured Cooldown for 'another-provider': {borgllm_specific.get_cooldown_duration('another-provider')}s"
        )
        print(
            f"  Configured Timeout for 'another-provider': {borgllm_specific.get_timeout_duration('another-provider')}s"
        )

        # Simulate calling a provider and hitting a timeout
        print("\n  Simulating calling 'test-provider-2' with a small timeout...\n")
        # To properly test timeout, you'd integrate with a mocked API client
        # For this example, we just show the configured value.

        # --- Example 3: Virtual Provider with Configured Timeout ---
        print("\n=== Example 3: Virtual Provider with Timeout and Cooldown ===")
        BorgLLM._instance = None  # Reset for fresh configuration
        BorgLLM._config_initialized = False

        # Configure a virtual provider with specific cooldown/timeout overrides
        cooldown_virtual_config = {"virtual-test-provider": 3, "default": 15}
        timeout_virtual_config = {"virtual-test-provider": 1.0, "default": 5.0}

        llm_virtual = create_llm(
            config_file=CONFIG_FILE,
            provider_name="virtual-test-provider",  # Defined in borg.yaml
            cooldown=cooldown_virtual_config,
            timeout=timeout_virtual_config,
        )
        borgllm_virtual = llm_virtual.borgllm_config

        print(
            f"  Virtual provider 'virtual-test-provider': cooldown={borgllm_virtual.get_cooldown_duration('virtual-test-provider')}s, timeout={borgllm_virtual.get_timeout_duration('virtual-test-provider')}s"
        )
        print(
            f"  Underlying 'test-provider-1' (upstream to virtual): cooldown={borgllm_virtual.get_cooldown_duration('test-provider-1')}s, timeout={borgllm_virtual.get_timeout_duration('test-provider-1')}s"
        )

        print(
            "  Simulating 429 on test-provider-1... Virtual provider should handle cooldown."
        )
        borgllm_virtual.signal_429("test-provider-1")

        # Attempt to get provider from virtual. It should either wait (if await is default/true)
        # or fall back to test-provider-2 if available, or raise error.
        # This example just demonstrates the configured timeouts/cooldowns being used.
        try:
            # If test-provider-1 is on cooldown, virtual-test-provider will attempt to find another upstream.
            # If only test-provider-1 is active and on cooldown, and allow_await_cooldown is True (default),
            # it would wait. If False, it would immediately raise an error.
            # We'll just demonstrate the values applied.
            pass  # No actual API call here to avoid complex mock setup for example
        except Exception as e:
            print(f"  (Expected) Error during virtual provider call: {e}")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print(
            "Please ensure you have necessary API keys (if using real providers) and try again."
        )


if __name__ == "__main__":
    main()
