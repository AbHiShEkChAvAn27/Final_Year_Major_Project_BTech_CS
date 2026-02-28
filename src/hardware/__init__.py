"""Hardware module - MCU registry, verification, and auto-detection"""

from .registry import MCU_HARDWARE_REGISTRY
from .verifier import hardware_constraint_verifier
from .detector import detect_all_boards, get_platform_from_boards, format_board_summary

__all__ = [
    "MCU_HARDWARE_REGISTRY",
    "hardware_constraint_verifier",
    "detect_all_boards",
    "get_platform_from_boards",
    "format_board_summary",
]
