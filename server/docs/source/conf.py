from __future__ import annotations

import os
import sys

from sphinx.ext.apidoc import main

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MCMRVLS"
copyright = "2026, artus9033"
author = "artus9033"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

here = os.path.abspath(os.path.dirname(__file__))
server_root = os.path.abspath(os.path.join(here, "..", ".."))
repo_root = os.path.abspath(os.path.join(server_root, ".."))
api_output_dir = os.path.join(here, "api")

sys.path.insert(0, repo_root)
sys.path.insert(0, server_root)


autodoc_mock_imports = [
    "cv2",
    "numpy",
    "torch",
    "torchvision",
    "imutils",
    "matplotlib",
    "PIL",
    "numba",
    "psutil",
    "rtsp",
    "pynput",
    "socketio",
    "flask",
    "werkzeug",
    "sklearn",
    "yaml",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "venv"]

html_theme = "sphinx_rtd_theme"

graphviz_output_format = "svg"
suppress_warnings = ["ref.python"]


def run_apidoc(_app):
    os.makedirs(api_output_dir, exist_ok=True)
    exclude_dirs = {
        "docs",
        "res",
        "venv",
        ".venv",
        "env",
        "build",
        "dist",
        "__pycache__",
    }
    exclude_paths = [os.path.join(server_root, name) for name in sorted(exclude_dirs)]
    main(
        [
            "-e",
            "-o",
            api_output_dir,
            "--force",
            "--remove-old",
            "--implicit-namespaces",
            "--module-first",
            server_root,
            *exclude_paths,
        ]
    )
    diagrams_path = os.path.join(api_output_dir, "diagrams.rst")
    diagram_targets = []
    package_name = os.path.basename(server_root)
    for root, dirs, files in os.walk(server_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            if filename in {"__init__.py", "__main__.py"}:
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as source_file:
                    source = source_file.read()
            except OSError:
                continue
            if "class " not in source:
                continue
            rel_path = os.path.relpath(file_path, server_root)
            module_path = rel_path[:-3].replace(os.sep, ".")
            diagram_targets.append(f"{package_name}.{module_path}")
    with open(diagrams_path, "w", encoding="utf-8") as diagrams_file:
        diagrams_file.write("Class Diagrams\n")
        diagrams_file.write("==============\n\n")
        if not diagram_targets:
            diagrams_file.write("No diagram targets found.\n")
            return
        for target in sorted(set(diagram_targets)):
            diagrams_file.write(f".. inheritance-diagram:: {target}\n")
            diagrams_file.write("   :parts: 2\n")
            diagrams_file.write("\n")


def setup(app):
    app.connect("builder-inited", run_apidoc)
