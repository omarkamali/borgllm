# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import toml

sys.path.insert(0, os.path.abspath("../../src"))

# Load version from pyproject.toml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pyproject_path = os.path.join(project_root, "pyproject.toml")
with open(pyproject_path, "r") as f:
    pyproject_content = toml.load(f)

project = "BorgLLM"
copyright = "2025, BorgLLM"
author = "Omar Kamali"
release = pyproject_content["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
