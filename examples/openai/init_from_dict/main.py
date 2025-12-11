"""
Initialize from Dictionary Example - BorgOpenAI

Shows how to initialize BorgOpenAI programmatically with a Python dictionary,
instead of loading from a borg.yaml file.
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 3. Run this example script: python main.py

load_dotenv()

# Define the configuration data as a dictionary
config_data = {
    "llm": {
        "providers": [
            {
                "name": "custom_api",
                "base_url": "http://localhost:8000/v1",
                "model": "custom-model",
                "api_key": os.getenv("OPENAI_API_KEY", "dummy-key"),
                "max_tokens": 8000,
            },
            {
                "name": "my_openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "max_tokens": 4096,
            },
        ],
        "default_model": "my_openai",
    }
}

# Initialize BorgOpenAI from a dictionary - no borg.yaml needed!
client = BorgOpenAI(initial_config_data=config_data)

# Verify the configuration
print("=== Dictionary-Initialized Configuration ===")

custom_api = client._resolve_provider("custom_api")
print(f"Provider 'custom_api':")
print(f"  Base URL: {custom_api.base_url}")
print(f"  Model: {custom_api.model}")
print(f"  Max Tokens: {custom_api.max_tokens}")
print()

my_openai = client._resolve_provider("my_openai")
print(f"Provider 'my_openai':")
print(f"  Base URL: {my_openai.base_url}")
print(f"  Model: {my_openai.model}")
print(f"  Max Tokens: {my_openai.max_tokens}")
print()

# Use the programmatically configured provider
if os.getenv("OPENAI_API_KEY"):
    print("Making API call with 'my_openai' provider...")
    response = client.chat.completions.create(
        model="my_openai",
        messages=[{"role": "user", "content": "Say 'Hello from dict config!' in 5 words."}],
    )
    print(f"Response: {response.choices[0].message.content}")
else:
    print("Set OPENAI_API_KEY to test actual API calls.")
