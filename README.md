# nmem

**nmem** is a Python-based framework for analyzing and simulating superconducting memory arrays. It includes modules for:

- **Data analysis** – Processing and visualizing experimental results.
- **Measurement automation** – Running and optimizing parameter sweeps.
- **Circuit simulation** – Simulating superconducting memory behavior using SPICE.

## Installation

To set up the environment using Conda, run:

    conda env create -f environment.yml
    conda activate nmem

## Repository Structure

    ├── LICENSE.txt
    ├── README.md
    ├── environment.yml
    ├── nmem.code-workspace
    ├── pyproject.toml
    └── src
        └── nmem
            ├── __init__.py
            ├── analysis/                     # Data analysis scripts
            ├── calculations/                 # Analytical models and calculations
            ├── layout/                       # Layout generation and visualization
            ├── measurement/                   # Experimental measurement scripts
            ├── simulation/                    # SPICE circuit simulations
            └── more subdirectories...

For a full breakdown, see the **Tree** section below.

## Dependency Installation (MariaDB)

If you need MariaDB for database-related operations, install the required packages:

    sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
    sudo apt-get install libmariadb-dev libmariadb-dev-compat

## License

This project is licensed under the MIT License – see [LICENSE.txt](LICENSE.txt) for details.

## Tree

├── LICENSE.txt
├── README.md
├── environment.yml
├── nmem.code-workspace
├── pyproject.toml
└── src
    └── nmem
        ├── analysis
        │   ├── analysis.py
        │   ├── array_parameter_plotting
        │   │   ├── array_parameter_plotting.py
        │   │   ├── main_analysis.py
        │   │   └── main_analysis_table.py
        │   ├── compare_C2C3_cells
        │   │   ├── SPG806_20241220_nMem_measure_enable_response_D6_A4_C2_2024-12-20 13-28-02.mat
        │   │   └── SPG806_20241220_nMem_measure_enable_response_D6_A4_C3_2024-12-20 17-28-57.mat
        │   ├── dc_sweep
        │   │   ├── data
        │   │   └── dc_sweep.py
        │   ├── enable_current_relation
        │   │   ├── data
        │   │   ├── data2
        │   │   └── enable_current_relation_fit.py
        │   ├── enable_current_relation_compare_C2C3_cells
        │   │   ├── compare_C2C3_cells.py
        │   │   └── data
        │   ├── enable_current_relation_v2
        │   │   ├── data
        │   │   ├── data2
        │   │   ├── enable_current_relation_temp.py
        │   │   └── enable_current_relation_v2.py
        │   ├── enable_write_current_sweep
        │   │   ├── data
        │   │   ├── data2
        │   │   ├── enable_write_current_sweep.py
        │   │   └── enable_write_sweep_fine.py
        │   ├── fitting_attempt.py
        │   ├── main_analysis.pdf
        │   ├── plot_branch_currents.py
        │   ├── plot_persistent_current.py
        │   ├── read_current_sweep_enable_read
        │   │   ├── data
        │   │   ├── data_290uA
        │   │   ├── data_300uA
        │   │   ├── data_310uA
        │   │   ├── data_310uA_C4
        │   │   ├── data_inverse
        │   │   ├── read_current_sweep_enable_read.py
        │   │   └── read_current_sweep_three.py
        │   ├── read_current_sweep_enable_write
        │   │   ├── data
        │   │   └── read_current_sweep_enable_write.py
        │   ├── read_current_sweep_enable_write_width
        │   │   ├── data
        │   │   └── read_current_sweep_enable_write_width.py
        │   ├── read_current_sweep_operating.py
        │   ├── read_current_sweep_operating_calculation.py
        │   ├── read_current_sweep_read_width
        │   │   ├── data
        │   │   └── read_current_sweep_read_width.py
        │   ├── read_current_sweep_write_current
        │   │   ├── data
        │   │   ├── data2
        │   │   └── write_current_read_sweep.py
        │   ├── read_current_sweep_write_current2
        │   │   ├── write_current_sweep.py
        │   │   ├── write_current_sweep_A2
        │   │   ├── write_current_sweep_B2_0
        │   │   ├── write_current_sweep_B2_1
        │   │   ├── write_current_sweep_B2_2
        │   │   ├── write_current_sweep_C2
        │   │   ├── write_current_sweep_C3
        │   │   ├── write_current_sweep_C3_2
        │   │   ├── write_current_sweep_C3_3
        │   │   └── write_current_sweep_C3_4
        │   ├── read_current_sweep_write_current3
        │   │   ├── data
        │   │   ├── data2
        │   │   ├── data3
        │   │   ├── data4
        │   │   ├── read_current_sweep_write_current.py
        │   │   ├── read_current_sweep_write_current_coarse.py
        │   │   ├── read_current_sweep_write_current_fine.py
        │   │   └── write_current_sweep_voltage_trace.py
        │   ├── read_current_sweep_write_width
        │   │   ├── data
        │   │   └── read_current_sweep_write_width.py
        │   ├── read_delay_v1
        │   │   ├── data
        │   │   ├── data2
        │   │   └── voltage_pulse_read_histogram.py
        │   ├── read_delay_v2
        │   │   ├── data
        │   │   ├── data2
        │   │   ├── data3
        │   │   ├── read_delay_read_current_sweep_fine.py
        │   │   └── read_delay_retention_test.py
        │   ├── state_currents.pdf
        │   ├── voltage_trace_emulate_slow
        │   │   ├── data
        │   │   └── voltage_trace_emulate_slow.py
        │   ├── write_current_sweep_enable_write
        │   │   ├── data
        │   │   └── write_current_sweep_enable_write.py
        │   ├── write_current_sweep_operation.py
        │   └── write_current_v_temp.py
        ├── calculations
        │   ├── analytical_model.py
        │   ├── calculations.py
        │   └── plotting.py
        ├── layout
        │   ├── die_layout_test.py
        │   ├── die_layout_v1.py
        │   ├── die_layout_v2.py
        │   └── sample
        ├── measurement
        │   ├── SPG806_config_ICE.yml
        │   ├── cells.py
        │   ├── functions.py
        │   ├── measure_delay_error.py
        │   ├── measure_enable_response.py
        │   ├── optimize.py
        │   ├── run_delay.py
        │   ├── run_optimal.py
        │   ├── run_optimize_parameter.py
        │   └── run_parameter_sweep.py
        └── simulation
            ├── geometry
            ├── nMem_test_1.py
            ├── nMem_test_2.py
            └── spice_circuits
                ├── functions.py
                ├── hTron_behavioral.asy
                ├── hTron_behavioral.lib
                ├── hTron_behavioral.log
                ├── nmem_cell_read.asc
                ├── nmem_cell_read.log
                ├── nmem_cell_read.net
                ├── nmem_cell_read.op.raw
                ├── nmem_cell_read.plt
                ├── nmem_cell_read.raw
                ├── nmem_cell_write.png
                ├── nmem_cell_write_slope.png
                ├── plotting.py
                ├── spice_data_plotting.py
                └── spice_simulation_raw
