"""Test package marker.

Without this file ``python -m unittest discover -s tests -t .`` fails with
``ImportError: Start directory is not importable``, and the suite can only be
run through a hand-written loader. Keeping ``tests`` an importable package lets
the whole suite run with one standard command from the project root.
"""
