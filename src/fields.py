#!/usr/bin/env python3
"""
Compatibility facade for the MOHID fields CLI.
"""

from .fields_cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
