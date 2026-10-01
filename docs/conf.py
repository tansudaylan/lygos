"""Sphinx configuration for the lygos documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "lygos"
author = "Tansu Daylan"
copyright = "2026, Tansu Daylan"
version = "0.2"
release = "0.2.0"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
# ecosystem and archive packages are not installed on Read the Docs, so autodoc imports stand-ins
autodoc_mock_imports = ["tdpy", "miletos", "pcat", "nicomedia", "h5py", "pandas", "astroquery", "s3fs", "tesswcs"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3}
htmlhelp_basename = "lygosdoc"
