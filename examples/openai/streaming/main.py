"""
Streaming Example - BorgOpenAI

Demonstrates streaming responses with both sync and async clients.
"""

import os
import asyncio
from dotenv import load_dotenv
from borgllm import BorgOpenAI, BorgAsyncOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 3. Run this example script: python main.py

load_dotenv()


def sync_streaming():
    """Demonstrate synchronous streaming."""
    print("=== Synchronous Streaming ===")
    
    client = BorgOpenAI()
    
    print("Streaming response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


async def async_streaming():
    """Demonstrate asynchronous streaming."""
    print("=== Asynchronous Streaming ===")
    
    client = BorgAsyncOpenAI()
    
    print("Streaming response: ", end="", flush=True)
    
    stream = await client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a short poem about coding (4 lines)."}],
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def streaming_with_virtual_provider():
    """Demonstrate streaming with a virtual provider."""
    print("=== Streaming with Virtual Provider ===")
    
    # Configure a virtual provider programmatically
    config_data = {
        "llm": {
            "providers": [
                {
                    "name": "primary",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "max_tokens": 1000,
                },
            ],
            "virtual": [
                {
                    "name": "my-virtual",
                    "upstreams": [{"name": "primary"}],
                }
            ],
        }
    }
    
    client = BorgOpenAI(initial_config_data=config_data)
    
    print("Streaming from virtual provider: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="my-virtual",
        messages=[{"role": "user", "content": "Say 'Hello from virtual provider!' word by word."}],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def main():
    """Run all streaming examples."""
    print("BorgOpenAI Streaming Examples")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment.")
        print("Please set your API key to run streaming examples.")
        return
    
    sync_streaming()
    asyncio.run(async_streaming())
    streaming_with_virtual_provider()
    
    print("All streaming examples completed!")


if __name__ == "__main__":
    main()
