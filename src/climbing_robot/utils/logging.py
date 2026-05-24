"""Logging setup with rich console output."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", use_rich: bool = True) -> None:
    """Configure root logger with rich or standard formatting."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if use_rich:
        try:
            from rich.logging import RichHandler  # noqa: PLC0415
            handler = RichHandler(rich_tracebacks=True, markup=True)
            fmt = "%(message)s"
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=[handler],
    )
