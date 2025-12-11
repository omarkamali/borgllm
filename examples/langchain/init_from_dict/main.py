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

# Define the configuration data as a dictionary
dict_config_data = {
    "llm": {
        "providers": [
            {
                "name": "custom_api",
                "base_url": "http://localhost:8000/v1",
                "model": "custom-model",
                "api_key": "${OPENAI_API_KEY}",
                "max_tokens": 8000,
            }
        ],
        "default_model": "custom_api",
    }
}

# Initialize BorgLLM from a dictionary
config_provider = BorgLLM.get_instance(initial_config_data=dict_config_data)

# Get the provider from the dictionary-initialized config
custom_api = config_provider.get("custom_api")

print(f"Retrieved Provider: {custom_api.name}")
print(f"Base URL: {custom_api.base_url}")
print(f"Model: {custom_api.model}")
print(f"Max Tokens: {custom_api.max_tokens}")
