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