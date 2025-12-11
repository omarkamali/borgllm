from .borgllm import BorgLLM
from .openai import BorgOpenAI, BorgAsyncOpenAI


# Lazy initialization function for set_default_provider
def set_default_provider(provider_name: str):
    """
    Sets the default LLM provider name for the global BorgLLM instance.

    This function creates or gets the global BorgLLM singleton instance and sets
    the default provider on it.
    """
    borgllm_instance = BorgLLM.get_instance()
    borgllm_instance.set_default_provider(provider_name)


# Lazy imports for optional LangChain integration
def __getattr__(name: str):
    """Lazy import for optional LangChain components."""
    if name in ("BorgLLMLangChainClient", "create_llm"):
        try:
            from .langchain import BorgLLMLangChainClient, create_llm
            if name == "BorgLLMLangChainClient":
                return BorgLLMLangChainClient
            return create_llm
        except ImportError as e:
            raise ImportError(
                f"LangChain integration requires the 'langchain' extra. "
                f"Install with: pip install borgllm[langchain]\n"
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BorgLLM",
    "set_default_provider",
    "BorgOpenAI",
    "BorgAsyncOpenAI",
    # Optional LangChain exports (require borgllm[langchain])
    "BorgLLMLangChainClient",
    "create_llm",
]
