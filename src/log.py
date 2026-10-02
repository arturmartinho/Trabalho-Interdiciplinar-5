import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# ANSI color codes
class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _emit(level: str, color: str, message: str) -> None:
    print(f"{_Colors.GRAY}[{_timestamp()}]{_Colors.RESET} "f"{color}{_Colors.BOLD}[{level}]{_Colors.RESET} {message}")

def success(message: str) -> None:
    """Log a success message."""
    _emit("SUCCESS", _Colors.GREEN, message)

def info(message: str) -> None:
    """Log an informational message."""
    _emit("INFO", _Colors.BLUE, message)

def debug(message: str) -> None:
    """Log a debug message (only on dev)."""
    if DEV_MODE:
        _emit("DEBUG", _Colors.CYAN, message)

def warning(message: str) -> None:
    """Log a warning message."""
    _emit("WARNING", _Colors.YELLOW, message)

def error(message: str) -> None:
    """Log an error message."""
    _emit("ERROR", _Colors.RED, message)
