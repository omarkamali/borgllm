## [2.0.0] - 2025-12-11

### Breaking Changes
- **LangChain is now optional**: Core install no longer includes LangChain dependencies. Install with `pip install borgllm[langchain]` for LangChain support
- `create_llm` and `BorgLLMLangChainClient` now require the `langchain` extra and will raise `ImportError` with install instructions if missing
- Examples reorganized into `examples/openai/` (core) and `examples/langchain/` (optional)

### Added
- **BorgOpenAI** and **BorgAsyncOpenAI** drop-in clients that duck-type the official OpenAI SDK, auto-configured via BorgLLM providers
- Support for OpenAI chat completions and the latest Responses API with streaming, cooldown, virtual providers, and multi-key rotation
- Shared `core` utilities module for reusable rate-limit handling and config resolution logic across integrations
- Comprehensive test suites for the new clients plus core utilities (35+ new tests)
- New examples demonstrating synchronous, asynchronous, multi-provider, virtual provider, and streaming usage

### Changed
- Core dependencies now only include OpenAI SDK (lighter install)
- LangChain dependencies moved to `[langchain]` optional extra
- README updated with new installation instructions and reorganized documentation

## [1.0.5] - 2025-11-08

### Added
- ZAI provider (`zai`) and its strong coding and agentic LLMs (`zai:zai/glm-4.6` and `zai:zai/glm-4.5-air`)
- Minimax provider (`minimax`) and its flagship model `minimax:minimax-m2`
- Omneity Labs provider (`omneity`) and its associated Moroccan-focused model (`omneity:sawalni-beta`)
- Update Moonshot AI provider (`moonshot`) to include new models (`moonshot:kimi-k2-0905-preview` and `moonshot:kimi-k2-thinking`)
- Trusted publishing via GitHub Actions with OIDC (`.github/workflows/publish.yml`).
- Upgraded publish script to a more robust Python implementation (`scripts/publish.py`).
- Increased testing coverage from Python 3.9 to 3.14.

### Fixed
- Changed name of Google provider's API key environment variable to `GOOGLE_API_KEY`