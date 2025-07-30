import os, sys
sys.path.insert(0, os.path.abspath('..'))  # so autodoc can find xsnow

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',      # for .md
    'myst_nb',          # for .ipynb
]

# myst_nb options:
nb_execution_mode = 'auto'   # run notebooks at build time
nb_execution_timeout = 120

# HTML theme
html_theme = 'sphinx_rtd_theme'
