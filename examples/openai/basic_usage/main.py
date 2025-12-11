"""
Basic Usage Example - BorgOpenAI

The simplest example demonstrating how to use BorgOpenAI with built-in providers.
No borg.yaml file required - works out of the box!
"""

import os
from dotenv import load_dotenv
from borgllm import BorgOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 3. Run this example script: python main.py

# Load environment variables from .env file
load_dotenv()

# Create a BorgOpenAI client - works out of the box with built-in providers!
client = BorgOpenAI()

# Simple chat completion using built-in 'openai' provider
# Model format: "provider:model" (e.g., "openai:gpt-4o")
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, BorgLLM with OpenAI client!"},
    ],
)

print(f"Response: {response.choices[0].message.content}")

# You can also use other built-in providers (check README for full list)
# Examples:
#   - "groq:llama-3.3-70b-versatile"
#   - "anthropic:claude-sonnet-4"
#   - "google:gemini-2.5-flash"
#   - "deepseek:deepseek-chat"
#   - "mistralai:mistral-large-latest"

# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And what about Japan?"},
]

chat_response = client.chat.completions.create(
    model="openai:gpt-4o-mini",
    messages=messages,
)
print(f"Chat Response: {chat_response.choices[0].message.content}")
