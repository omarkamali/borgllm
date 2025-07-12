import os
from dotenv import load_dotenv
from borgllm import BorgLLM

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

# Test virtual provider with token approximation
# This demonstrates how BorgLLM selects the best available provider based on approximate_tokens.

qwen_auto_5k = config_provider.get("qwen-auto", approximate_tokens=5000)
print(f"Retrieved Qwen Auto (5k tokens): {qwen_auto_5k.name}")
print(f"Max Tokens: {qwen_auto_5k.max_tokens}")

qwen_auto_100k = config_provider.get("qwen-auto", approximate_tokens=100000)
print(f"Retrieved Qwen Auto (100k tokens): {qwen_auto_100k.name}")
print(f"Max Tokens: {qwen_auto_100k.max_tokens}")
