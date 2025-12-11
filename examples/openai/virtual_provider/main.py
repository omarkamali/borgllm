"""
Virtual Provider Example - BorgOpenAI

Demonstrates virtual providers that automatically fall back between multiple
upstream LLM providers. When one provider hits rate limits or is unavailable,
BorgLLM seamlessly switches to the next available upstream.
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    GROQ_API_KEY=your_groq_key_here
#    CEREBRAS_API_KEY=your_cerebras_key_here
# 3. Run this example script: python main.py

load_dotenv()

# BorgOpenAI reads borg.yaml which defines a virtual provider "qwen-auto"
# that falls back: groq -> cerebras -> local_qwen
client = BorgOpenAI()

print("=== Virtual Provider Example ===")
print("The 'qwen-auto' virtual provider will automatically select the best")
print("available upstream provider based on availability and rate limits.\n")

# Get the default provider (qwen-auto is set as default in borg.yaml)
default_provider = client._resolve_provider("qwen-auto")

print(f"Resolved Provider: {default_provider.name}")
print(f"Base URL: {default_provider.base_url}")
print(f"Model: {default_provider.model}")
print()

# The virtual provider resolves to the first available upstream
# In this case, it should resolve to groq:qwen/qwen3-32b if GROQ_API_KEY is set

# Make an actual API call using the virtual provider
if os.getenv("GROQ_API_KEY") or os.getenv("CEREBRAS_API_KEY"):
    print("Making API call with virtual provider 'qwen-auto'...")
    response = client.chat.completions.create(
        model="qwen-auto",
        messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
    )
    print(f"Response: {response.choices[0].message.content}")
else:
    print("Set GROQ_API_KEY or CEREBRAS_API_KEY to test actual API calls.")
    print("The virtual provider will automatically select the available upstream.")
