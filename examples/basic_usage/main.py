import os
from dotenv import load_dotenv
from borgllm import create_llm
from langchain_core.messages import HumanMessage, AIMessage

# --- Instructions --- #
# 1. Ensure you have 'uv' installed (pip install uv).
# 2. Run 'uv pip install' in the project root to install dependencies.
# 3. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 4. Run this example script: python3 main.py

# Load environment variables from .env file
load_dotenv()

# Create a LangChain LLM client using BorgLLM's built-in providers.
# This uses the built-in 'openai:gpt-4o' provider - works out of the box, no borg.yaml needed!
llm_client = create_llm("openai:gpt-4o")

# Check out all the available providers in README.md

# Invoke the LLM client with a simple message.
response = llm_client.invoke("Hello, LangChain with BorgLLM!")

# Print the response content.
print(f"LLM Response: {response.content}")

# You can also use it with a list of messages, typical for chat models:
messages = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="Paris."),
    HumanMessage(content="And what about Japan?"),
]

chat_response = llm_client.invoke(messages)
print(f"Chat Response: {chat_response.content}")
