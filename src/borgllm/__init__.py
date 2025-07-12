from .borgllm import BorgLLM
from .langchain import BorgLLMLangChainClient, create_llm


# Lazy initialization function for set_default_provider
def set_default_provider(provider_name: str):
    """
    Sets the default LLM provider name for the global BorgLLM instance.

    This function creates or gets the global BorgLLM singleton instance and sets
    the default provider on it.
    """
    borgllm_instance = BorgLLM.get_instance()
    borgllm_instance.set_default_provider(provider_name)


__all__ = [
    "BorgLLM",
    "BorgLLMLangChainClient",
    "create_llm",
    "set_default_provider",
]
