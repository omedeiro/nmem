"""SPICE circuit simulation package for nmem."""

from . import core
from . import waveform
from . import utils
from . import config

__version__ = "0.1.0"

# Convenience imports for common functionality
from .waveform import generate_memory_protocol_sequence, WaveformGenerator
from .core import LTspiceRunner, SimulationAutomator
from .utils import save_pwl_file, get_data_dir, get_results_dir
from .config.settings import get_default_config, ConfigManager
