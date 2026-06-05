# Contributing to Sentimental

Thank you for your interest in contributing! We appreciate all kinds of contributions — bug reports, feature requests, documentation improvements, and code changes.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## Code of Conduct

This project adheres to a standard open-source code of conduct. By participating, you agree to:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community

---

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behaviour vs actual behaviour
- Screenshots or logs if applicable
- Your environment (OS, Python version, etc.)

### 2. Suggesting Features

Feature requests are welcome! Please open an issue with:
- A clear description of the feature
- Why it would be useful
- Any relevant examples or mockups

### 3. Code Contributions

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and test them locally
4. Commit with a clear, descriptive message
5. Push to your fork and open a Pull Request

---

## Development Setup

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/sentimental.git
cd sentimental

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (handled automatically on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Run the app locally
streamlit run app.py
```

---

## Project Structure

```
sentimental/
├── app.py              # Streamlit UI
├── train_model.py      # Training pipeline & dataset loaders
├── preprocess.py       # NLTK preprocessing
├── utils.py            # Model I/O, prediction, chart helpers
├── requirements.txt    # Dependencies
├── render.yaml         # Render deployment config
├── Procfile            # Web deployment
├── setup.sh            # Streamlit Cloud startup
├── .streamlit/
│   └── config.toml     # Theme & server settings
├── models/             # Trained model artifacts (gitignored)
└── train.csv           # Primary dataset (optional)
```

---

## Coding Standards

- **Python version:** 3.9+
- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/)
- **Line length:** 88 characters (Black formatter standard)
- **Type hints:** Use type annotations for all new functions and methods
- **Docstrings:** Write clear docstrings following Google or NumPy style
- **Logging:** Use the `logging` module instead of `print()`
- **Imports:** Group standard library → third-party → local imports, separated by blank lines

### Before Submitting a PR

```bash
# Format code (if using black)
black *.py

# Lint
flake8 *.py

# Run the app to verify everything works
streamlit run app.py
```

---

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code refactoring (no feature or bug fix) |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks, dependency updates |

Examples:
```
feat: add VADER sentiment features to training pipeline
fix: handle empty CSV uploads gracefully
docs: update README with deployment instructions
refactor: simplify preprocessing tokenisation logic
```

---

## Areas Where You Can Help

- **Model improvements:** Experiment with different algorithms (SVM, Naive Bayes, fine-tuned transformers)
- **Preprocessing:** Improve text cleaning, add spell-checking, handle emojis better
- **UI/UX:** Enhance the Streamlit interface, add dark/light mode toggle
- **Testing:** Write unit tests for preprocessing and prediction functions
- **Documentation:** Fix typos, add examples, improve explanations
- **Deployment:** Add Docker support, CI/CD pipelines, monitoring
- **Datasets:** Integrate additional labeled sentiment datasets

---

## Getting Help

- Open an issue for bug reports or feature requests
- Check existing issues and discussions before opening a new one
- Be patient and respectful — maintainers are volunteers

---

Thank you for contributing! 🎉
