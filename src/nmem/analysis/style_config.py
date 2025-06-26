#!/usr/bin/env python3
"""
Style configuration utility for nmem plotting scripts.

This module provides easy access to set global plotting styles across all scripts.
"""

from nmem.analysis.styles import apply_global_style, get_style_mode, set_style_mode


def set_presentation_style():
    """Set global style to presentation mode."""
    set_style_mode("presentation")


def set_paper_style():
    """Set global style to paper/publication mode."""
    set_style_mode("paper")


def set_thesis_style():
    """Set global style to thesis mode."""
    set_style_mode("thesis")


def current_style():
    """Get the current global style mode."""
    return get_style_mode()


def apply_style():
    """Apply the currently configured global style."""
    apply_global_style()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python style_config.py {presentation|paper|thesis}")
        print(f"Current style: {current_style()}")
        sys.exit(1)

    style = sys.argv[1].lower()
    valid_styles = ["presentation", "pres", "paper", "publication", "thesis"]

    if style not in valid_styles:
        print(f"Invalid style '{style}'. Valid options: {valid_styles}")
        sys.exit(1)

    set_style_mode(style)
    print(f"Global style set to: {current_style()}")
