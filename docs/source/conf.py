# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

master_doc = 'index'
autodoc_mock_imports = ["scipy", "pandas" ,"matplotlib", "submodlib", "tqdm", "numpy","torch", "sklearn", "utils_jl", "models", "tensorflow", "snorkel", 
"checkmate", "checkpoints", "data_feeder_utils", "data_types",  
"gen_cross_entropy_utils", "pr_utils", "test", "train", "utils"] #config

# -- Project information -----------------------------------------------------

project = 'SPEAR(DECILE)'
copyright = '2021 DECILE'
author = 'Ayush_Maheshwari, Guttu_Sai_Abhishek, Harshad_Ingole, Parth_Laturia, Vineeth_Dorna'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    #'numpydoc'
    'sphinxcontrib.napoleon',
    'sphinxcontrib.bibtex'
# 'rinoh.frontend.sphinx'
]

bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' #'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
html_static_path = []

#below line makes functions to print in order they were defined in files. added by abhishek
autodoc_member_order = 'bysource'