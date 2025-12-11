"""
Provider Cooldown Example - BorgOpenAI

Demonstrates BorgLLM's built-in 429 (Too Many Requests) error handling,
including signaling a cooldown for a provider and how virtual providers
intelligently avoid blocked upstreams.
"""

import os
import time
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    GROQ_API_KEY=your_groq_key_here
#    CEREBRAS_API_KEY=your_cerebras_key_here
# 3. Run this example script: python main.py

load_dotenv()

# Initialize BorgOpenAI with borg.yaml that defines virtual providers
client = BorgOpenAI()

print("=== Provider Cooldown Example ===\n")

# Simulate a 429 error by manually signaling cooldown
# In real usage, BorgOpenAI automatically handles 429 responses
provider_to_block = "groq:qwen/qwen3-32b"
cooldown_duration = 5  # seconds

print(f"Simulating 429 error on '{provider_to_block}'...")
client._borgllm_config.signal_429(provider_to_block, duration=cooldown_duration)
print(f"Provider '{provider_to_block}' is now blocked for {cooldown_duration} seconds.\n")

# Attempt to get the blocked provider directly (should fail)
print("Attempting to get blocked provider directly...")
try:
    client._resolve_provider(provider_to_block)
    print("  Unexpected: Provider was available!")
except ValueError as e:
    print(f"  Expected error: {e}\n")

# Virtual provider should automatically avoid the blocked upstream
# In borg.yaml, qwen-auto uses: groq -> cerebras -> local_qwen
# When groq is blocked, it should use cerebras instead
print("Getting virtual provider 'qwen-auto' (should avoid blocked upstream)...")
try:
    resolved = client._resolve_provider("qwen-auto")
    print(f"  Resolved to: {resolved.name}")
    print(f"  (Expected to avoid {provider_to_block})\n")
except ValueError as e:
    print(f"  Error: {e}\n")

# Wait for cooldown to expire
print(f"Waiting {cooldown_duration + 1} seconds for cooldown to expire...")
time.sleep(cooldown_duration + 1)

# Attempt to get the provider after cooldown (should succeed)
print("\nAttempting to get provider after cooldown...")
try:
    resolved = client._resolve_provider(provider_to_block)
    print(f"  Success! Retrieved: {resolved.name}")
    print("  (Cooldown has expired)")
except ValueError as e:
    print(f"  Unexpected error: {e}")

print("\n=== Automatic 429 Handling ===")
print("In real usage, BorgOpenAI automatically:")
print("1. Detects 429 responses from the API")
print("2. Signals cooldown for the affected provider")
print("3. Retries with exponential backoff")
print("4. Falls back to alternative providers if available")
