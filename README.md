# nmem

**nmem** is a comprehensive Python-based framework for analyzing, simulating, and fabricating superconducting nanowire memory (nMem) arrays. This research tool supports the complete development cycle of superconducting memory devices from design to characterization.

## Key Features

### 🔬 **Data Analysis & Visualization**

- **Experimental data processing** – Import and analyze measurement data from various sources
- **Advanced plotting suite** – Generate publication-ready plots for bit error rate (BER), current sweeps, state currents, and array parameters
- **Statistical analysis** – Process alignment statistics, memory retention data, and error analysis
- **Matrix visualization** – Create parameter matrices and resistance maps for device arrays

### ⚡ **Measurement Automation**

- **Parameter sweeps** – Automated current, voltage, and timing parameter optimization
- **BER testing** – Comprehensive bit error rate characterization across operating conditions
- **Real-time optimization** – Bayesian optimization for device parameter tuning
- **Instrument control** – Integration with measurement equipment via qnnpy

### 🧮 **Modeling & Simulation**

- **Analytical models** – Physics-based models for superconducting nanowire behavior
- **SPICE simulations** – Circuit-level simulations using behavioral hTron models
- **Temperature dependencies** – Critical current and state current calculations
- **Read/write margin analysis** – Operating window optimization

### 🏗️ **Layout & Fabrication**

- **GDS layout generation** – Automated mask design using gdspy and phidl
- **Device arrays** – Scalable memory cell and test structure layouts
- **Process integration** – Support for e-beam lithography and fabrication workflows
- **Test structure libraries** – Standard cells for characterization

## Installation

### Environment Setup

Set up the conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate nmem
```

### Dependencies

Key packages include:

- **Analysis**: numpy, scipy, pandas, matplotlib, scikit-optimize
- **Layout**: gdspy, phidl, qnngds
- **Simulation**: ltspice, tdgl
- **Measurement**: pyvisa, qnnpy, mariadb

## Quick Start

### Data Analysis

```python
from nmem.analysis import import_directory, plot_ber_sweep
from nmem.scripts import plot_array_fidelity_bar

# Import experimental data
data_dict = import_directory("path/to/data")[0]

# Generate analysis plots
plot_array_fidelity_bar.main(save_dir="plots/")
```

### Device Simulation

```python
from nmem.calculations import analytical_model
from nmem.simulation.spice_circuits import functions

# Run analytical model
model_results = analytical_model.run_analysis()

# SPICE circuit simulation
spice_data = functions.process_read_data(ltspice_file)
```

### Layout Generation

```python
from nmem.layout import die_layout_v2

# Generate memory array layout
memory_array = die_layout_v2.nMemArray8()
memory_array.write_gds("memory_array.gds")
```

## Repository Structure

```text
├── LICENSE.txt                      # MIT License
├── README.md                        # This file
├── environment.yml                  # Conda environment specification
├── pyproject.toml                   # Python project configuration
└── src/nmem/                       # Main source code
    ├── analysis/                    # Data analysis and visualization
    │   ├── core_analysis.py         # Core analysis functions
    │   ├── data_import.py           # Data import utilities
    │   ├── plotting.py              # General plotting functions
    │   ├── matrix_plots.py          # Array parameter visualization
    │   ├── sweep_plots.py           # Parameter sweep plotting
    │   ├── state_currents_plots.py  # State current analysis
    │   ├── bit_error.py             # BER analysis functions
    │   └── styles.py                # Plot styling and configuration
    ├── calculations/                # Analytical models and calculations
    │   ├── analytical_model.py      # Physics-based device models
    │   ├── calculations.py          # Core calculation functions
    │   └── plotting.py              # Model visualization
    ├── data/                        # Experimental data storage
    │   ├── ber_*/                   # Bit error rate measurements
    │   ├── dc_sweep*/               # DC characterization data
    │   ├── voltage_*/               # Voltage measurements
    │   └── wafer_*/                 # Wafer-level test data
    ├── layout/                      # GDS layout generation
    │   ├── die_layout_v*.py         # Die layout generators
    │   ├── circuit_*.tex            # Circuit schematic generation
    │   └── sample/                  # Layout examples
    ├── measurement/                 # Experimental measurement
    │   ├── functions.py             # Measurement utilities
    │   ├── cells.py                 # Device parameter database
    │   ├── run_*.py                 # Measurement scripts
    │   └── optimize.py              # Parameter optimization
    ├── plots/                       # Generated plots and figures
    ├── scripts/                     # Analysis and plotting scripts
    │   ├── plot_*.py                # Individual plot generators
    │   └── calc_*.py                # Calculation scripts
    └── simulation/                  # Device simulation
        ├── spice_circuits/          # SPICE simulation files
        │   ├── *.asc                # LTspice circuit files
        │   ├── functions.py         # Simulation utilities
        │   └── plotting.py          # Simulation visualization
        └── geometry/                # Device geometry models
```

## Key Modules

### Analysis (`nmem.analysis`)

**Core Functions:**

- `core_analysis.py` - Data processing, BER calculation, fitting algorithms
- `data_import.py` - Import measurement data from various formats (.mat, .csv)
- `sweep_plots.py` - Comprehensive parameter sweep visualization
- `matrix_plots.py` - Array parameter matrices and die maps

**Specialized Analysis:**

- `bit_error.py` - Bit error rate analysis and statistics
- `state_currents_plots.py` - Memory state current characterization
- `alignment_plots.py` - E-beam alignment analysis
- `histogram_utils.py` - Statistical data analysis

### Measurement (`nmem.measurement`)

**Automation:**

- `run_parameter_sweep.py` - Automated parameter sweeps
- `run_optimize_parameter.py` - Bayesian optimization of device parameters
- `measure_enable_response.py` - Enable current characterization
- `run_delay.py` - Memory retention measurements

**Device Database:**

- `cells.py` - Comprehensive device parameter database with 16 characterized cells
- Includes critical currents, resistances, operating points, and BER data

### Simulation (`nmem.simulation`)

**SPICE Models:**

- Behavioral hTron models for superconducting nanowires
- Complete memory cell circuits with read/write operations
- Parameter sweeps and Monte Carlo analysis

**Analytical Models:**

- Temperature-dependent critical current models
- State current calculations for memory operations
- Read/write margin analysis

### Layout (`nmem.layout`)

**Device Generation:**

- Memory array layouts (4×4, 8×8 configurations)
- Test structure libraries
- Automated routing and interconnects
- GDS export for fabrication

## Advanced Features

### Measurement Automation

- **Real-time optimization** using scikit-optimize
- **Multi-parameter sweeps** with early termination
- **Statistical analysis** with outlier rejection
- **Database integration** for data management

### Analysis Capabilities

- **BER vs. operating parameters** with confidence intervals
- **Array uniformity analysis** across dies and wafers
- **Temperature coefficient extraction** from measurements
- **Critical current distribution mapping**

### Visualization Suite

- **Publication-ready plots** with configurable styles
- **Interactive parameter exploration** tools
- **3D bar charts** for multi-dimensional data
- **Matrix plots** for array visualization

## Dependencies and MariaDB

If using database features, install MariaDB dependencies:

```bash
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
sudo apt-get install libmariadb-dev libmariadb-dev-compat
```

## Recent Updates

This framework has been significantly expanded from its original scope to include:

- **Comprehensive device characterization** with 16 fully characterized memory cells
- **Advanced plotting capabilities** with 50+ specialized plot types
- **Automated measurement workflows** with real-time optimization
- **Complete fabrication support** from layout to device testing
- **SPICE simulation integration** with behavioral device models

## Contributing

This is a research tool under active development. Key areas for contribution:

- **New analysis methods** for superconducting memory characterization  
- **Additional device models** for different nanowire geometries
- **Measurement automation** for new test setups
- **Visualization improvements** for data presentation

## Citation

If you use this framework in your research, please cite the associated publications on superconducting nanowire memory devices.

## License

This project is licensed under the MIT License – see [LICENSE.txt](LICENSE.txt) for details.
