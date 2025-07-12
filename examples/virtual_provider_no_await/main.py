import os
from dotenv import load_dotenv
from borgllm import BorgLLM
import time

# --- Instructions --- #
# 1. Ensure you have 'uv' installed (pip install uv).
# 2. Run 'uv pip install' in the project root to install dependencies.
# 3. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 4. Run this example script: python3 main.py

# Load environment variables from .env file
load_dotenv()

# Define a temporary configuration for this specific scenario.
# This config includes a virtual provider with a single upstream.
temp_dict_config = {
    "llm": {
        "providers": [
            {
                "name": "awaitable_real_provider",
                "base_url": "https://api.awaitable.com",
                "model": "awaitable-model",
                "api_key": "${OPENAI_API_KEY}",
                "max_tokens": 1000,
            }
        ],
        "virtual": [
            {
                "name": "single_upstream_virtual",
                "upstreams": [{"name": "awaitable_real_provider"}],
            }
        ],
        "default_model": "single_upstream_virtual",
    }
}

# Initialize BorgLLM with the temporary dictionary configuration.
await_config_provider = BorgLLM.get_instance(initial_config_data=temp_dict_config)

# Block the single upstream of the virtual provider.
await_config_provider.signal_429("awaitable_real_provider", duration=5)
print("Single upstream 'awaitable_real_provider' blocked for 5 seconds.")

# Attempt to get the virtual provider without awaiting cooldown. Expect it to fail immediately.
# This demonstrates that if allow_await_cooldown is False, BorgLLM will not wait for cooldown to expire.
print(
    "Attempting to get 'single_upstream_virtual' with allow_await_cooldown=False (should fail immediately):"
)
start_time_no_await = time.time()
try:
    await_config_provider.get("single_upstream_virtual", allow_await_cooldown=False)
except ValueError as e:
    end_time_no_await = time.time()
    elapsed_time_no_await = end_time_no_await - start_time_no_await
    print(f"Caught expected error: {e} (Elapsed: {elapsed_time_no_await:.4f}s)")
