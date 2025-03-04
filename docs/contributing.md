# Contributing to Pypulate

Thank you for considering contributing to Pypulate! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to Pypulate:

1. **Report bugs**: If you find a bug, please create an issue on GitHub with a detailed description of the problem, including steps to reproduce it.

2. **Suggest features**: If you have an idea for a new feature or improvement, please create an issue on GitHub to discuss it.

3. **Contribute code**: If you want to contribute code, please follow the steps below.

## Development Setup

1. Fork the repository on GitHub.

2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/pypulate.git
   cd pypulate
   ```

3. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. We use the following tools to enforce code style:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting

You can run these tools with:
```bash
black src tests
isort src tests
flake8 src tests
```

### Documentation

- All functions, classes, and modules should have docstrings following the NumPy docstring format.
- Update the documentation when adding or modifying features.
- Run the documentation locally to check your changes:
  ```bash
  mkdocs serve
  ```

### Testing

- Write tests for all new features and bug fixes.
- Make sure all tests pass before submitting a pull request:
  ```bash
  pytest
  ```

## Pull Request Process

1. Update the documentation with details of changes to the interface, if applicable.
2. Update the tests to cover your changes.
3. Make sure all tests pass.
4. Submit a pull request to the `main` branch.
5. The pull request will be reviewed by maintainers, who may request changes or improvements.
6. Once approved, your pull request will be merged.

## Adding New KPIs or Moving Averages

If you want to add a new KPI or moving average function:

1. Add the function to the appropriate module (`kpi/business_kpi.py` for KPIs, `moving_averages/movingaverages.py` for moving averages).
2. Write comprehensive docstrings with parameters, return values, and examples.
3. Add tests for the new function.
4. Update the documentation to include the new function.
5. Add the function to the appropriate `__init__.py` file to expose it.

## License

By contributing to Pypulate, you agree that your contributions will be licensed under the project's MIT License. 