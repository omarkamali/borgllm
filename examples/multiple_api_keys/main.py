import os
from dotenv import load_dotenv
from borgllm import BorgLLM

# --- Instructions --- #
# 1. Ensure you have 'uv' installed (pip install uv).
# 2. Run 'uv pip install' in the project root to install dependencies.
# 3. Create a '.env' file in this directory with your API keys. For multiple keys, use comma-separated values:
#    OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
# 4. Run this example script: python3 main.py

# Load environment variables from .env file
load_dotenv()

# Initialize the BorgLLM config provider.
# Built-in providers like 'openai' automatically pick up keys from environment variables (e.g., OPENAI_API_KEYS).
config_provider = BorgLLM.get_instance()

print("\n--- BorgLLM Multiple API Keys Example (Built-in Provider) ---")
print("Demonstrating round-robin API key usage for 'openai:gpt-4o'\n")

# Retrieve the 'openai:gpt-4o' provider multiple times to see key rotation
for i in range(5):
    provider = config_provider.get("openai:gpt-4o")
    # Display only the last 5 characters of the key for security
    print(f"Call {i+1}: Using API Key ending in ...{provider.api_key[-5:]}")
