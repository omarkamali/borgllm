BorgLLM Documentation
=====================

**BorgLLM** is a universal Python LLM client with integrated providers, automatic API key rotation,
rate limit handling, and configurable provider fallback strategies.

.. note::
   **v2.0.0**: LangChain is now optional. Core install provides ``BorgOpenAI`` and ``BorgAsyncOpenAI``
   drop-in replacements for the OpenAI SDK. Install ``borgllm[langchain]`` for LangChain support.

Installation
------------

.. code-block:: bash

   # Core install (OpenAI SDK integration)
   pip install borgllm

   # With optional LangChain support
   pip install borgllm[langchain]

Quick Start
-----------

**BorgOpenAI (Drop-in OpenAI SDK replacement)**

.. code-block:: python

   from borgllm import BorgOpenAI

   client = BorgOpenAI()
   response = client.chat.completions.create(
       model="openai:gpt-5.2",  # or any provider:model
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response.choices[0].message.content)

**Async Usage**

.. code-block:: python

   from borgllm import BorgAsyncOpenAI

   client = BorgAsyncOpenAI()
   response = await client.chat.completions.create(
       model="anthropic:claude-opus-4-5",
       messages=[{"role": "user", "content": "Hello!"}]
   )

**LangChain Integration (Optional)**

.. code-block:: python

   # Requires: pip install borgllm[langchain]
   from borgllm import create_llm

   llm = create_llm("openai:gpt-5.2", temperature=0.7)
   response = llm.invoke("Hello!")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 