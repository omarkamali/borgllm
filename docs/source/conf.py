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
copyright = "2025, BorgLLM - Omar Kamali"
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

# Furo theme options - align with website color scheme
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#00cc66",
        "color-brand-content": "#00cc66",
        "color-admonition-background": "rgba(0, 204, 102, 0.1)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00ff7f",
        "color-brand-content": "#00ff7f",
        "color-background-primary": "#0a0a0a",
        "color-background-secondary": "#111111",
        "color-admonition-background": "rgba(0, 255, 127, 0.1)",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_title = "BorgLLM"
html_short_title = "BorgLLM"
