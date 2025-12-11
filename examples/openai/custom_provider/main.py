"""
Custom Provider Example - BorgOpenAI

Demonstrates how to configure and use a custom LLM provider defined in borg.yaml.
Useful for local models (Ollama, LM Studio) or custom API endpoints.
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here (or your custom API key)
# 3. Optionally start a local model server (Ollama, LM Studio, etc.)
# 4. Run this example script: python main.py

load_dotenv()

# BorgOpenAI will automatically read borg.yaml from the current directory
# The borg.yaml defines a custom provider "local_gemma" pointing to localhost
client = BorgOpenAI()

# Get provider info to verify configuration
provider_config = client._resolve_provider("local_gemma")

print("=== Custom Provider Configuration ===")
print(f"Provider Name: {provider_config.name}")
print(f"Base URL: {provider_config.base_url}")
print(f"Model: {provider_config.model}")
print(f"Max Tokens: {provider_config.max_tokens}")
print()

# Use the custom provider for chat completion
# Note: This will fail if you don't have a local server running
# Uncomment the code below to test with a real local server

# response = client.chat.completions.create(
#     model="local_gemma",  # Use the provider name from borg.yaml
#     messages=[
#         {"role": "user", "content": "Hello from BorgOpenAI!"},
#     ],
# )
# print(f"Response: {response.choices[0].message.content}")

print("To test with a real local server:")
print("1. Start Ollama or LM Studio on localhost:1234")
print("2. Uncomment the API call code above")
print("3. Run this script again")
