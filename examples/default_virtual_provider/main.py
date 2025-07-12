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

# Get the default LLM provider (qwen-auto, which is virtual according to borg.yaml)
default_model = config_provider.get()

print(f"Retrieved Default Model: {default_model.name}")
print(f"Base URL: {default_model.base_url}")
print(f"Model: {default_model.model}")
print(f"Max Tokens: {default_model.max_tokens}")
