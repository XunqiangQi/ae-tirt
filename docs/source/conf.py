import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "AE-TIRT"
author = "Anonymous Author(s)"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []
language = "en"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
