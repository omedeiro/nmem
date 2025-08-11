"""File I/O utilities for SPICE simulation."""

import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np


def save_pwl_file(filename: Union[str, Path], t: np.ndarray, i: np.ndarray) -> None:
    """Save time and current arrays to a PWL (Piecewise Linear) file for LTspice.

    Args:
        filename: Output filename or path
        t: Time array
        i: Current array
    """
    with open(filename, "w") as f:
        for time, current in zip(t, i):
            f.write(f"{time:.12e}\t{current:.12e}\n")


def read_pwl_file(filename: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read time and current arrays from a PWL file.

    Args:
        filename: Input filename or path

    Returns:
        Tuple of (time_array, current_array)
    """
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_data_dir() -> Path:
    """Get the data directory for storing simulation results."""
    data_dir = (
        get_project_root() / "src" / "nmem" / "simulation" / "spice_circuits" / "data"
    )
    return ensure_directory_exists(data_dir)


def get_results_dir() -> Path:
    """Get the results directory for storing processed results."""
    results_dir = (
        get_project_root()
        / "src"
        / "nmem"
        / "simulation"
        / "spice_circuits"
        / "results"
    )
    return ensure_directory_exists(results_dir)


def get_waveforms_dir() -> Path:
    """Get the waveforms directory for storing PWL files."""
    waveforms_dir = (
        get_project_root()
        / "src"
        / "nmem"
        / "simulation"
        / "spice_circuits"
        / "data"
        / "waveforms"
    )
    return ensure_directory_exists(waveforms_dir)


def get_circuit_files_dir() -> Path:
    """Get the circuit files directory."""
    return (
        get_project_root()
        / "src"
        / "nmem"
        / "simulation"
        / "spice_circuits"
        / "circuit_files"
    )


def get_config_dir() -> Path:
    """Get the configuration directory."""
    config_dir = (
        get_project_root() / "src" / "nmem" / "simulation" / "spice_circuits" / "config"
    )
    return ensure_directory_exists(config_dir)
