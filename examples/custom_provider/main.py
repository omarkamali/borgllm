import os
from dotenv import load_dotenv
from borgllm import BorgLLM

# --- Instructions --- #
# 1. Ensure you have 'uv' installed (pip install uv).
# 2. Run 'uv pip install' in the project root to install dependencies.
# 3. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 4. Run this example script: python3 main.py

# Load environment variables from .env file
load_dotenv()

# Initialize the config provider. It will read from borg.yaml in this directory.
config_provider = BorgLLM.get_instance()

# Get a custom LLM provider defined in borg.yaml
local_gemma = config_provider.get("local_gemma")

print(f"Retrieved Provider: {local_gemma.name}")
print(f"Base URL: {local_gemma.base_url}")
print(f"Model: {local_gemma.model}")
print(f"Max Tokens: {local_gemma.max_tokens}")
