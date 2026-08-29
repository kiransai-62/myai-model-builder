# 🤝 Contributing to MYAI

Thank you for your interest in contributing to **MYAI**! We welcome bug reports, documentation improvements, feature proposals, and community discussions.

---

## 🧭 Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please treat everyone with respect and empathy.

---

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/myai.git
cd myai

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[train,eval,serving]"
```

---

## 🧪 Running Tests

Before submitting any pull request, ensure all tests pass:

```bash
pytest -v
```

---

## 📝 Pull Request Guidelines

1. **Keep Changes Focused**: Each pull request should address a single feature, bug fix, or documentation update.
2. **Follow Code Style**: Maintain clean, type-annotated Python code following PEP 8.
3. **Add Tests**: Include unit tests for any new features or bug fixes.
4. **Update Documentation**: If your change adds or modifies CLI commands or options, update [`CLI.md`](file:///e:/CMD%20GitHub/myai/CLI.md) and [`README.md`](file:///e:/CMD%20GitHub/myai/README.md).
