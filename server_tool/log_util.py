"""
Simple logging utilities for retargeting server.

Provides basic logging setup without external dependencies.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    from termcolor import colored
except ImportError:

    def colored(text, *args, **kwargs):
        return text


# root logger
_root_logger = logging.getLogger()
_console_handler: Optional[logging.Handler] = None


def log_init():
    """Initialize root logger with DEBUG level."""
    _root_logger.setLevel(logging.DEBUG)


def enable_console(formatter=None):
    """Enable console logging with colored output."""
    global _root_logger, _console_handler

    if _console_handler is not None:
        return

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    if formatter is None:
        formatter = ConsoleFormatter()
    _console_handler.setFormatter(formatter)
    _root_logger.addHandler(_console_handler)


def disable_console():
    """Disable console logging."""
    global _root_logger, _console_handler

    if _console_handler is None:
        return

    _root_logger.removeHandler(_console_handler)
    _console_handler = None


class Formatter(logging.Formatter):
    """Base logging formatter."""

    time_str = "[%(asctime)s]"
    level_str = "[%(levelname)s]"
    msg_str = "%(message)s"
    src_str = "(%(name)s @ %(filename)s:%(lineno)d)"


class ConsoleFormatter(Formatter):
    """Colored console formatter."""

    COLOR_MAP = {
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def format(self, record: logging.LogRecord) -> str:
        time_str = colored(self.time_str, "blue")
        level_color = self.COLOR_MAP.get(record.levelno, "white")
        level_str = colored(self.level_str, level_color)
        src_str = colored(self.src_str, "dark_grey")

        fmt_str = f"{time_str} {level_str} {self.msg_str} {src_str}"
        formatter = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
