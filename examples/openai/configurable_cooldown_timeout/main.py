#!/usr/bin/env python3
"""
Configurable Cooldown and Timeout Example - BorgOpenAI

Demonstrates how to configure custom cooldown periods and request timeouts
globally, for specific providers, or for specific provider:model combinations.
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI, BorgLLM

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 3. Run this example script: python main.py

load_dotenv()

# Path to the borg.yaml for this example
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "borg.yaml")


def example_global_config():
    """Example 1: Global cooldown and timeout configuration."""
    print("=== Example 1: Global Cooldown (10s) and Timeout (5.0s) ===")
    
    # Reset singleton for fresh configuration
    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    
    # Create client with global cooldown of 10 seconds and timeout of 5.0 seconds
    client = BorgOpenAI(
        config_file=CONFIG_FILE,
        cooldown=10,      # 10 seconds cooldown for all providers
        timeout=5.0,      # 5.0 seconds timeout for all operations
    )
    
    borgllm = client._borgllm_config
    
    print(f"  Cooldown for 'test-provider-1': {borgllm.get_cooldown_duration('test-provider-1')}s")
    print(f"  Timeout for 'test-provider-1': {borgllm.get_timeout_duration('test-provider-1')}s")
    print(f"  Cooldown for 'any_other_provider': {borgllm.get_cooldown_duration('any_other_provider')}s")
    print(f"  Timeout for 'any_other_provider': {borgllm.get_timeout_duration('any_other_provider')}s")
    
    # Simulate a 429 error and observe cooldown
    print("  Simulating 429 error on 'test-provider-1'...")
    borgllm.signal_429("test-provider-1")
    print(f"  Is 'test-provider-1' unusable? {borgllm._is_provider_unusable('test-provider-1')}")
    print()


def example_provider_specific_config():
    """Example 2: Provider-specific cooldown and timeout configuration."""
    print("=== Example 2: Provider-Specific Cooldown and Timeout ===")
    
    # Reset singleton for fresh configuration
    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    
    # Configure different cooldowns/timeouts for different providers
    cooldown_config = {
        "test-provider-2": 5,    # Shorter cooldown for this specific provider
        "openai": 60,            # Longer cooldown for OpenAI
        "default": 15,           # Default for others
    }
    timeout_config = {
        "test-provider-2": 2.0,  # Shorter timeout
        "openai": 10.0,          # Longer timeout for OpenAI
        "default": 7.0,          # Default for others
    }
    
    client = BorgOpenAI(
        config_file=CONFIG_FILE,
        cooldown=cooldown_config,
        timeout=timeout_config,
    )
    
    borgllm = client._borgllm_config
    
    print(f"  Cooldown for 'test-provider-2': {borgllm.get_cooldown_duration('test-provider-2')}s")
    print(f"  Timeout for 'test-provider-2': {borgllm.get_timeout_duration('test-provider-2')}s")
    print(f"  Cooldown for 'openai:gpt-4o': {borgllm.get_cooldown_duration('openai:gpt-4o')}s")
    print(f"  Timeout for 'openai:gpt-4o': {borgllm.get_timeout_duration('openai:gpt-4o')}s")
    print(f"  Cooldown for 'another-provider': {borgllm.get_cooldown_duration('another-provider')}s")
    print(f"  Timeout for 'another-provider': {borgllm.get_timeout_duration('another-provider')}s")
    print()


def example_virtual_provider_config():
    """Example 3: Virtual provider with configured timeout and cooldown."""
    print("=== Example 3: Virtual Provider with Timeout and Cooldown ===")
    
    # Reset singleton for fresh configuration
    BorgLLM._instance = None
    BorgLLM._config_initialized = False
    
    # Configure cooldown/timeout for virtual provider and its upstreams
    cooldown_config = {
        "virtual-test-provider": 3,
        "default": 15,
    }
    timeout_config = {
        "virtual-test-provider": 1.0,
        "default": 5.0,
    }
    
    client = BorgOpenAI(
        config_file=CONFIG_FILE,
        cooldown=cooldown_config,
        timeout=timeout_config,
    )
    
    borgllm = client._borgllm_config
    
    print(f"  Virtual 'virtual-test-provider': cooldown={borgllm.get_cooldown_duration('virtual-test-provider')}s, timeout={borgllm.get_timeout_duration('virtual-test-provider')}s")
    print(f"  Upstream 'test-provider-1': cooldown={borgllm.get_cooldown_duration('test-provider-1')}s, timeout={borgllm.get_timeout_duration('test-provider-1')}s")
    
    print("  Simulating 429 on 'test-provider-1'...")
    borgllm.signal_429("test-provider-1")
    print("  Virtual provider will now use fallback upstream if available.")
    print()


def main():
    """Run all examples."""
    print("BorgOpenAI Configurable Cooldown and Timeout Example")
    print("=" * 55)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nNote: OPENAI_API_KEY not found in environment.")
        print("This example demonstrates configuration only.\n")
    
    example_global_config()
    example_provider_specific_config()
    example_virtual_provider_config()
    
    print("=== Summary ===")
    print("Cooldown and timeout can be configured:")
    print("  - Globally: cooldown=10, timeout=5.0")
    print("  - Per-provider: cooldown={'openai': 60, 'default': 15}")
    print("  - Per-model: cooldown={'openai:gpt-4o': 30}")


if __name__ == "__main__":
    main()
