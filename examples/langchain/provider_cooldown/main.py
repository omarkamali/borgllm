import os
from dotenv import load_dotenv
from borgllm import BorgLLM
import time

# --- Instructions --- #
# 1. Ensure you have 'uv' installed (pip install uv).
# 2. Run 'uv pip install' in the project root to install dependencies.
# 3. Create a '.env' file in this directory with your API keys:
#    GROQ_API_KEY=your_groq_key_here
#    CEREBRAS_API_KEY=your_cerebras_key_here
# 4. Run this example script: python3 main.py

# Load environment variables from .env file
load_dotenv()

# Initialize the config provider. It will read from borg.yaml in this directory.
config_provider = BorgLLM.get_instance()

# Test 429 signaling and cooldown
# This demonstrates how to signal a 429 (Too Many Requests) for a provider
# and how BorgLLM handles cooldown periods.

provider_to_block = "groq:qwen/qwen3-32b"
cooldown_duration = 5  # seconds
config_provider.signal_429(provider_to_block, duration=cooldown_duration)
print(f"Provider '{provider_to_block}' is now blocked for {cooldown_duration} seconds.")

# Attempt to get the blocked provider directly (should fail)
try:
    config_provider.get(provider_to_block)
except ValueError as e:
    print(f"Caught expected error when getting blocked provider: {e}")

# Attempt to get a virtual provider that might use the blocked provider (should avoid it)
# In borg.yaml, qwen-auto is virtual and can use groq or cerebras.
# When groq is blocked, qwen-auto should use cerebras.
try:
    qwen_auto_after_429 = config_provider.get("qwen-auto", approximate_tokens=5000)
    print(
        f"Retrieved: {qwen_auto_after_429.name} (Expected to avoid {provider_to_block})"
    )
except ValueError as e:
    print(f"Caught unexpected error: {e}")

print(f"Waiting {cooldown_duration + 1} seconds for cooldown to expire...")
time.sleep(cooldown_duration + 1)

# Attempt to get the provider after cooldown (should succeed)
try:
    groq_after_cooldown = config_provider.get(provider_to_block)
    print(f"Retrieved: {groq_after_cooldown.name} (Cooldown expired)")
except ValueError as e:
    print(f"Caught unexpected error after cooldown: {e}")
