"""Core functionality for SPICE circuit simulation."""

from .data_processing import (
    get_persistent_current,
    get_write_current,
    get_max_output,
    get_processed_state_currents,
    process_data_dict_sweep,
    process_data_dict_write_sweep,
    get_write_sweep_data,
    get_bit_error_rate,
    get_switching_probability,
    get_current_or_voltage,
    safe_max,
    safe_min,
)

from .ltspice_interface import (
    LTspiceRunner,
    SimulationAutomator,
    load_ltspice_data,
)
