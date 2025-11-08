## [1.0.5] - 2025-11-08

### Added
- Omneity Labs provider (`omneity`) and its associated Moroccan-focused models (`omneity:sawalni-beta`)
- Update Moonshot AI provider (`moonshot`) to include new models (`moonshot:kimi-k2-0905-preview` and `moonshot:kimi-k2-thinking`)
- Trusted publishing via GitHub Actions with OIDC (`.github/workflows/publish.yml`).
- Upgraded publish script to a more robust Python implementation (`scripts/publish.py`).
- Increased testing coverage from Python 3.8 to 3.14.

### Fixed
- Changed name of Google provider's API key environment variable to `GOOGLE_API_KEY`