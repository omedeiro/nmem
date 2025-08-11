# SPICE Simulation Automation

A streamlined system for automated SPICE simulation with waveform generation and analysis plotting.

## Overview

This system provides **three main functions** for SPICE simulation automation:

1. **`generate_waveforms()`** - Create simulation input waveforms
2. **`run_simulation()`** - Execute SPICE simulation 
3. **`plot_results()`** - Generate analysis plots

## Quick Start

### 1. Create Configuration File

```bash
# Create default configuration
python spice_simulation.py --create-config simulation_config.yaml
```

### 2. Run Complete Workflow

```bash
# Run all steps (waveforms → simulation → plotting)
python spice_simulation.py --config simulation_config.yaml
```

### 3. Run Individual Steps

```bash
# Generate waveforms only
python spice_simulation.py --config simulation_config.yaml --step waveforms

# Run simulation only
python spice_simulation.py --config simulation_config.yaml --step simulation

# Generate plots only  
python spice_simulation.py --config simulation_config.yaml --step plotting
```

## Programmatic Usage

```python
from nmem.simulation.spice_circuits.spice_simulation import (
    generate_waveforms, 
    run_simulation, 
    plot_results,
    load_config
)

# Load configuration
config = load_config('simulation_config.yaml')

# Step 1: Generate waveforms
waveform_files = generate_waveforms(config)

# Step 2: Run simulation
simulation_results = run_simulation(config, waveform_files)

# Step 3: Generate plots
plot_files = plot_results(config, simulation_results)
```

## Configuration

The system is configured via YAML files. See `config/simulation_config.yaml` for a complete example.

### Key Configuration Sections

#### Waveforms
```yaml
waveforms:
  standard:
    cycle_time: 1.0e-6
    write_amplitude: 80.0e-6
    read_amplitude: 725.0e-6
    # ... other parameters
```

#### Simulation
```yaml
simulation:
  template: 'nmem_cell_read_template.cir'
  ltspice_path: '/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe'
  parameters:
    temp: 4.2
    timeout: 300
```

#### Plotting
```yaml
plotting:
  types: ['transient', 'multi_panel']
  generate_report: true
  style: 'publication'
```

## File Organization

```
spice_circuits/
├── spice_simulation.py              # Main interface
├── config/
│   ├── simulation_config.yaml       # Default configuration
│   └── plotting_config.yaml         # Plot styling config
├── core/
│   ├── ltspice_interface.py         # SPICE automation
│   ├── unified_plotter.py           # Plotting system
│   └── data_processing.py           # Data utilities
├── scripts/
│   └── unified_waveform_generator.py # Waveform generation
├── circuit_files/                   # SPICE circuit templates
├── data/                            # Generated waveforms
└── results/                         # Simulation outputs and plots
```

## Core Functions

### 1. Generate Waveforms

Creates SPICE-compatible waveform files for simulation input.

**Features:**
- Standard protocol sequences
- Parameter sweeps
- Waveform comparisons
- Configurable timing and amplitudes

### 2. Run Simulation

Executes LTspice simulations with generated waveforms.

**Features:**
- Automated netlist generation
- Cross-platform LTspice integration (WSL support)
- Parameter sweeps
- Error handling and logging

### 3. Plot Results

Generates professional analysis plots from simulation data.

**Features:**
- Multiple plot types (transient, current sweep, voltage output)
- Multi-panel analysis views
- Configurable styling (publication, presentation, technical)
- Comprehensive report generation
- Multiple export formats (PNG, PDF, SVG)

## Advanced Usage

### Custom Waveforms

```yaml
waveforms:
  sweep:
    parameter: 'read_amplitude'
    values: [700e-6, 725e-6, 750e-6]
    base_config: 'standard'
```

### Parameter Sweeps

```yaml
simulation:
  sweep:
    parameter: 'temp'
    values: [4.0, 4.2, 4.5]
    output_prefix: 'temp_sweep'
```

### Custom Plot Types

```yaml
plotting:
  types: 
    - 'transient'
    - 'current_sweep' 
    - 'voltage_output'
    - 'multi_panel'
```

## Output Structure

```
results/
├── simulation.raw               # Raw simulation data
├── simulation.cir              # Generated netlist
├── plots/
│   ├── standard_transient.png
│   ├── standard_multi_panel.png
│   └── comprehensive_report/
└── sweep_results/              # Parameter sweep outputs
```

## Dependencies

- **Python 3.8+**
- **LTspice** (Windows/WSL)
- **ltspice** (Python library)
- **matplotlib** (plotting)
- **numpy** (numerical)
- **pyyaml** (configuration)

## Troubleshooting

### Common Issues

1. **LTspice path**: Update `ltspice_path` in config for your system
2. **WSL paths**: The system automatically handles Windows/Linux path conversion
3. **Missing data**: Check waveform generation completed successfully
4. **Plot errors**: Verify simulation produced valid output data

### Logging

Set logging level for debugging:
```bash
python spice_simulation.py --config config.yaml --log-level DEBUG
```

## Migration from Legacy Scripts

This system replaces the following legacy files:
- `demo_*.py` scripts → `spice_simulation.py`
- Individual plotting scripts → `unified_plotter.py`
- Multiple waveform generators → `unified_waveform_generator.py`

The new system provides:
- **Unified interface** instead of scattered scripts
- **Configuration-driven** instead of hardcoded parameters  
- **Modular functions** instead of monolithic demos
- **Professional output** instead of test plots
