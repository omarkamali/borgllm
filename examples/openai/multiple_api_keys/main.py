"""
Multiple API Keys Example - BorgOpenAI

Shows how to configure and use multiple API keys for a provider,
demonstrating the automatic round-robin rotation of keys.
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with multiple API keys (comma-separated):
#    OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
# 3. Run this example script: python main.py

load_dotenv()

print("=== BorgOpenAI Multiple API Keys Example ===")
print("Demonstrating round-robin API key rotation for 'openai:gpt-4o'\n")

# Initialize BorgOpenAI - it automatically picks up OPENAI_API_KEYS from environment
client = BorgOpenAI()

# Retrieve the provider multiple times to see key rotation
# Each call rotates to the next API key in the list
for i in range(5):
    provider = client._resolve_provider("openai:gpt-4o")
    # Display only the last 5 characters of the key for security
    key_suffix = provider.api_key[-5:] if len(provider.api_key) > 5 else provider.api_key
    print(f"Call {i+1}: Using API Key ending in ...{key_suffix}")

print()
print("Note: If you only have one key, you'll see the same key each time.")
print("With multiple keys (OPENAI_API_KEYS=key1,key2,key3), they rotate round-robin.")
print()

# Make actual API calls to demonstrate rotation in practice
if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEYS"):
    print("Making 3 API calls to demonstrate key rotation...")
    for i in range(3):
        response = client.chat.completions.create(
            model="openai:gpt-4o-mini",
            messages=[{"role": "user", "content": f"Say 'Call {i+1}' and nothing else."}],
        )
        print(f"Response {i+1}: {response.choices[0].message.content}")
else:
    print("Set OPENAI_API_KEY or OPENAI_API_KEYS to test actual API calls.")
