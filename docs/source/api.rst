.. _api:

API Reference
=============

This section provides detailed documentation for the BorgLLM API.

.. automodule:: borgllm.borgllm
   :members: BorgLLM
   :undoc-members:
   :show-inheritance:

.. automodule:: borgllm.langchain
   :undoc-members:
   :show-inheritance:

.. autoclass:: borgllm.langchain.BorgLLMLangChainClient
   :members:
   :special-members: __init__
   :exclude-members: _generate, _agenerate, _handle_rate_limit_error, _handle_non_rate_limit_error, _get_fresh_config_and_update, _update_config_from_provider

.. autofunction:: borgllm.langchain.create_llm

.. note::
   The `BorgLLMLangChainClient` class is a low-level component and is typically not manually initialized by users. It is primarily used internally by the `create_llm` convenience function to manage LLM configurations and handle rate limiting. 