"""Configuration management for SPICE simulations."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, asdict
from ..utils.constants import *


@dataclass
class WaveformConfig:
    """Configuration for waveform generation."""

    cycle_time: float = DEFAULT_CYCLE_TIME
    pulse_sigma: float = DEFAULT_PULSE_SIGMA
    hold_width_write: float = DEFAULT_HOLD_WIDTH_WRITE
    hold_width_read: float = DEFAULT_HOLD_WIDTH_READ
    hold_width_clear: float = DEFAULT_HOLD_WIDTH_CLEAR
    write_amplitude: float = DEFAULT_WRITE_AMPLITUDE
    read_amplitude: float = DEFAULT_READ_AMPLITUDE
    enab_write_amplitude: float = DEFAULT_ENAB_WRITE_AMPLITUDE
    enab_read_amplitude: float = DEFAULT_ENAB_READ_AMPLITUDE
    clear_amplitude: float = DEFAULT_CLEAR_AMPLITUDE
    dt: float = DEFAULT_DT


@dataclass
class SimulationConfig:
    """Configuration for simulation settings."""

    ltspice_path: str = DEFAULT_LTSPICE_PATH
    template_netlist: str = "nmem_cell_read_v3.cir"
    output_format: str = "csv"
    logging_level: str = "INFO"


@dataclass
class PathConfig:
    """Configuration for file paths."""

    data_dir: Optional[str] = None
    results_dir: Optional[str] = None
    circuit_files_dir: Optional[str] = None
    templates_dir: Optional[str] = None


@dataclass
class PlottingConfig:
    """Configuration for plotting settings."""

    figure_size: tuple = (10, 6)
    dpi: int = 300
    font_size: int = 12
    line_width: float = 1.5
    grid: bool = True
    style: str = "default"


@dataclass
class SpiceConfig:
    """Main configuration class containing all settings."""

    waveform: WaveformConfig = None
    simulation: SimulationConfig = None
    paths: PathConfig = None
    plotting: PlottingConfig = None

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.waveform is None:
            self.waveform = WaveformConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.plotting is None:
            self.plotting = PlottingConfig()


class ConfigManager:
    """Manager for loading and saving configuration files."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> SpiceConfig:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file (.yaml or .json)

        Returns:
            Loaded configuration object
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load data based on file extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Convert to config objects
        config = SpiceConfig()

        if "waveform" in data:
            config.waveform = WaveformConfig(**data["waveform"])
        if "simulation" in data:
            config.simulation = SimulationConfig(**data["simulation"])
        if "paths" in data:
            config.paths = PathConfig(**data["paths"])
        if "plotting" in data:
            config.plotting = PlottingConfig(**data["plotting"])

        return config

    @staticmethod
    def save_config(config: SpiceConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            config: Configuration object to save
            config_path: Path where to save configuration
        """
        config_path = Path(config_path)

        # Convert to dictionary
        data = {
            "waveform": asdict(config.waveform),
            "simulation": asdict(config.simulation),
            "paths": asdict(config.paths),
            "plotting": asdict(config.plotting),
        }

        # Save based on file extension
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @staticmethod
    def create_default_config(config_path: Union[str, Path]) -> SpiceConfig:
        """Create and save a default configuration file.

        Args:
            config_path: Path where to save the default configuration

        Returns:
            Default configuration object
        """
        config = SpiceConfig()
        ConfigManager.save_config(config, config_path)
        return config

    @staticmethod
    def get_config(config_path: Optional[Union[str, Path]] = None) -> SpiceConfig:
        """Get configuration, loading from file if provided, otherwise return default.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Configuration object
        """
        if config_path is not None:
            return ConfigManager.load_config(config_path)
        else:
            return SpiceConfig()


# Convenience function for getting default config
def get_default_config() -> SpiceConfig:
    """Get default configuration."""
    return SpiceConfig()
