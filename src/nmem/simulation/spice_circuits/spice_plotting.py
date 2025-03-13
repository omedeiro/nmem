import matplotlib.pyplot as plt
import numpy as np
import ltspice
from nmem.simulation.spice_circuits.nmem_cell_plot import (
    plot_nmem_cell,
    plot_read_current_output,
    plot_tran_data,
)
from nmem.simulation.spice_circuits.spice_data_processing import (
    import_csv_dir,
    process_data_dict_sweep,
    get_write_sweep_data,
)


def figure_plot_write_read_clear() -> None:
    file_path = "spice_simulation_raw/nmem_cell_write_read_clear.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file

    fig, ax = plt.subplots()
    ax = plot_tran_data(ax, l, "Ix(HR:drain)")
    ax = plot_tran_data(ax, l, "V(ichr)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    plt.show()


def figure_plot_enable_read_sweep() -> None:

    data_dict = import_csv_dir("spice_simulation_raw/read_current_sweep/")
    fig, ax = plt.subplots()
    for key, data in data_dict.items():
        new_dict = {"read_current": data[:, 0], "read_output": data[:, 1:]}
        ax = plot_read_current_output(ax, new_dict)
    enable_read_currents, zero_currents, one_currents = process_data_dict_sweep(
        data_dict
    )

    fig, ax = plt.subplots()
    ax.plot(enable_read_currents, zero_currents, "-o", label="Read 0")
    ax.plot(enable_read_currents, one_currents, "-o", label="Read 1")
    ax.set_ylabel("Read Current (uA)")
    ax.set_xlabel("Enable Read Current (uA)")
    read_margin = zero_currents - one_currents
    optimal_idx = np.argmax(read_margin)

    ax.plot(
        enable_read_currents[optimal_idx],
        zero_currents[optimal_idx],
        "x",
        label="Optimal Read Margin",
    )
    optimal_read = (
        np.array(zero_currents[optimal_idx]) - np.array(read_margin[optimal_idx]) / 2
    )
    ax.plot(
        enable_read_currents[optimal_idx],
        optimal_read,
        "x",
        label="Optimal Read Margin",
    )
    print(f"Optimal Read: {optimal_read}")
    ax.legend()
    plt.show()


def figure_plot_write_sweep_data(
    file_path: str = "spice_simulation_raw/write_current_sweep/",
):
    data_dict = get_write_sweep_data(file_path)
    fig, ax = plt.subplots()
    for key, data in data_dict.items():
        data_array = data["data"]
        new_dict = {"read_current": data_array[:, 0], "read_output": data_array[:, 1:]}
        plot_read_current_output(ax, new_dict)

    write_current_list = []
    persistent_current_list = []
    read_margin_list = []
    for key, data in data_dict.items():
        write_current_list.append(key)
        persistent_current_list.append(data["persistent_current"])
        read_margin_list.append(data["read_margin"])

    fig, ax = plt.subplots()
    ax.plot(
        write_current_list, persistent_current_list, "-o", label="Persistent Current"
    )
    ax.plot(write_current_list, read_margin_list, "-o", label="Read Margin")
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Current (uA)")
    ax.legend()
    plt.show()


def figure_plot_all_write_sweeps() -> None:
    fig, ax = plt.subplots()
    plot_nmem_cell(ax, "spice_simulation_raw/nmem_cell_write_200uA.raw")
    plt.show()
    fig, ax = plt.subplots()
    plot_nmem_cell(ax, "spice_simulation_raw/nmem_cell_write_1000uA.raw")
    plt.show()
    fig, ax = plt.subplots()
    plot_nmem_cell(ax, "spice_simulation_raw/nmem_cell_write_step_5uA.raw")
    plt.show()
    fig, ax = plt.subplots()
    plot_nmem_cell(ax, "spice_simulation_raw/nmem_cell_write_step_10uA.raw")
    plt.show()


if __name__ == "__main__":
    figure_plot_write_read_clear()
    figure_plot_enable_read_sweep()
    figure_plot_write_sweep_data()
    figure_plot_all_write_sweeps()
