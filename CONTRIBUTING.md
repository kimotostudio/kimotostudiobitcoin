# Contributing to Bitcoin Bottom Detector

Thank you for your interest in contributing!

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/kimotostudio/kimotostudiobitcoin/issues)
2. Create a new issue using the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots (if applicable)
   - Environment details

### Suggesting Features

1. Check [existing feature requests](https://github.com/kimotostudio/kimotostudiobitcoin/issues?q=is%3Aissue+label%3Aenhancement)
2. Create a new issue using the feature request template
3. Describe:
   - The problem it solves
   - Proposed solution
   - Alternative solutions considered

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit (`git commit -m '[ADD] Amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Standards

- **Python:** PEP 8 style guide
- **Commit messages:** `[TYPE] Brief description` (types: ADD, FIX, UPDATE, REFACTOR, DOCS, CONFIG)
- **Documentation:** Add docstrings for new functions
- **Stats:** Every data-processing function should return a stats dict

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/kimotostudiobitcoin.git
cd kimotostudiobitcoin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Questions?

Feel free to [open an issue](https://github.com/kimotostudio/kimotostudiobitcoin/issues).
