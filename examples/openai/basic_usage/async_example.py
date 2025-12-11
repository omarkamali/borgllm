"""
Async Basic Usage Example - BorgAsyncOpenAI

Demonstrates async usage with concurrent requests and streaming.
"""

import os
import asyncio
from dotenv import load_dotenv
from borgllm import BorgAsyncOpenAI

# --- Instructions --- #
# 1. pip install borgllm
# 2. Create a '.env' file in this directory with your API keys:
#    OPENAI_API_KEY=your_openai_key_here
# 3. Run this example script: python async_example.py

load_dotenv()


async def simple_async_call():
    """Simple async chat completion."""
    print("=== Simple Async Call ===")
    
    client = BorgAsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
    )
    
    print(f"Response: {response.choices[0].message.content}\n")


async def concurrent_requests():
    """Make multiple concurrent LLM requests."""
    print("=== Concurrent Requests ===")
    
    client = BorgAsyncOpenAI()
    
    questions = [
        "What is the capital of France? Reply in one word.",
        "What is the capital of Japan? Reply in one word.",
        "What is the capital of Brazil? Reply in one word.",
    ]
    
    async def ask(question: str) -> str:
        response = await client.chat.completions.create(
            model="openai:gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content
    
    print("Asking 3 questions concurrently...")
    results = await asyncio.gather(*[ask(q) for q in questions])
    
    for question, answer in zip(questions, results):
        print(f"Q: {question}")
        print(f"A: {answer}\n")


async def async_streaming():
    """Demonstrate async streaming."""
    print("=== Async Streaming ===")
    
    client = BorgAsyncOpenAI()
    
    print("Streaming response: ", end="", flush=True)
    
    stream = await client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a haiku about programming."}],
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


async def main():
    """Run all async examples."""
    await simple_async_call()
    await concurrent_requests()
    await async_streaming()
    print("All async examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
