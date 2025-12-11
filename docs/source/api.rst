.. _api:

API Reference
=============

This section provides detailed documentation for the BorgLLM API.

Core Classes
------------

BorgLLM Configuration
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: borgllm.borgllm
   :members: BorgLLM
   :undoc-members:
   :show-inheritance:

OpenAI SDK Clients
~~~~~~~~~~~~~~~~~~

The primary way to use BorgLLM. These classes are drop-in replacements for the official OpenAI SDK.

.. automodule:: borgllm.openai
   :members: BorgOpenAI, BorgAsyncOpenAI
   :undoc-members:
   :show-inheritance:

LangChain Integration (Optional)
--------------------------------

.. note::
   Requires ``pip install borgllm[langchain]``

.. automodule:: borgllm.langchain
   :undoc-members:
   :show-inheritance:

.. autoclass:: borgllm.langchain.BorgLLMLangChainClient
   :members:
   :special-members: __init__
   :exclude-members: _generate, _agenerate, _handle_rate_limit_error, _handle_non_rate_limit_error, _get_fresh_config_and_update, _update_config_from_provider

.. autofunction:: borgllm.langchain.create_llm

.. note::
   The ``BorgLLMLangChainClient`` class is a low-level component typically not manually initialized.
   Use the ``create_llm`` convenience function instead. 