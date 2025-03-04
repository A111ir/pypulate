# Installation

## Prerequisites

Pypulate requires:

- Python 3.7 or higher
- NumPy 1.20.0 or higher

## Installing from PyPI

The easiest way to install Pypulate is using pip:

```bash
pip install pypulate
```

This will install Pypulate and all its dependencies.

## Installing from Source

If you want to install the latest development version, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/yourusername/pypulate.git
```

## Development Installation

For development purposes, you can clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/pypulate.git
cd pypulate
pip install -e .
```

This will install the package in development mode, allowing you to modify the code and see the changes immediately without reinstalling.

## Verifying Installation

You can verify that Pypulate is installed correctly by importing it in Python:

```python
import pypulate
print(pypulate.__version__)
```

This should print the version number of Pypulate. 