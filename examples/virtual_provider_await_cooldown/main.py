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

# Block the single upstream of the virtual provider to simulate a 429.
await_config_provider.signal_429("awaitable_real_provider", duration=5)
print("Single upstream 'awaitable_real_provider' blocked for 5 seconds.")

# Attempt to get the virtual provider with a short timeout, expecting it to fail.
# This demonstrates that if the timeout is shorter than the cooldown, it will fail.
short_timeout = 2  # seconds
print(
    f"Attempting to get 'single_upstream_virtual' with timeout={short_timeout}s (should fail):"
)
start_time_timeout = time.time()
try:
    await_config_provider.get("single_upstream_virtual", timeout=short_timeout)
except ValueError as e:
    end_time_timeout = time.time()
    elapsed_time = end_time_timeout - start_time_timeout
    print(f"Caught expected timeout error: {e} (Elapsed: {elapsed_time:.2f}s)")

# Block the upstream again for the next test.
await_config_provider.signal_429("awaitable_real_provider", duration=5)
print(
    "Single upstream 'awaitable_real_provider' blocked again for 5 seconds for the next test."
)

# Attempt to get the virtual provider, allowing it to await cooldown. Expect it to succeed after cooldown.
# This demonstrates the default behavior where BorgLLM waits for a provider to become available.
print("\nAttempting to get 'single_upstream_virtual' (should await cooldown):")
start_time_await = time.time()
try:
    resolved_after_await = await_config_provider.get(
        "single_upstream_virtual", allow_await_cooldown=True
    )
    end_time_await = time.time()
    elapsed_time_await = end_time_await - start_time_await
    print(
        f"Retrieved: {resolved_after_await.name} after awaiting cooldown (Elapsed: {elapsed_time_await:.2f}s)"
    )
except ValueError as e:
    print(f"Caught unexpected error during await: {e}")
