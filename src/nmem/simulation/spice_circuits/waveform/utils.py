"""Backward-compatible waveform utilities wrapper."""

# Import all functionality from the new generators module
from .generators import (
    flat_top_gaussian,
    generate_memory_protocol_sequence,
    WaveformGenerator,
)

# Import file I/O utilities
from ..utils.file_io import save_pwl_file

# Re-export for backward compatibility
__all__ = [
    "flat_top_gaussian",
    "generate_memory_protocol_sequence",
    "save_pwl_file",
    "WaveformGenerator",
]
