"""Alias legado: ``python -m src.loadbalancer`` equivale a ``python -m src serve``."""

import sys

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
