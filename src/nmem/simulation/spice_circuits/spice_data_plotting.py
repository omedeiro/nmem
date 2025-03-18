import ltspice
import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import CMAP
from nmem.simulation.spice_circuits.functions import (
    import_raw_dir,
    process_read_data,
)
from nmem.simulation.spice_circuits.plotting import (
    plot_current_sweep_ber,
    plot_current_sweep_output,
    plot_current_sweep_persistent,
    plot_tran_data,
)


def example_write_read_clear() -> None:
    file_path = "spice_simulation_raw/nmem_cell_write_read_clear.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()

    fig, ax = plt.subplots()
    ax = plot_tran_data(ax, l, "Ix(HR:drain)")
    ax = plot_tran_data(ax, l, "V(ichr)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    plt.show()


def example_step_read_sweep_write() -> None:

    data_dict = import_raw_dir("spice_simulation_raw/read_current_sweep_2/")
    fig, ax = plt.subplots()
    colors = CMAP(np.linspace(0, 1, len(data_dict)))
    for i, (key, l) in enumerate(data_dict.items()):
        write_current = key[-9:-6]
        processed_data = process_read_data(l)
        ax = plot_current_sweep_output(
            ax, processed_data, color=colors[i], label="Write " + write_current + "uA"
        )
    ax.legend(loc="upper left", ncol=1, prop={"size": 12}, bbox_to_anchor=(1.05, 1))


def figure_plot_all_write_sweeps() -> None:
    fig, ax = plt.subplots()
    plot_tran_data(ax, "spice_simulation_raw/nmem_cell_write_200uA.raw", "I(ichr)")
    plt.show()
    # fig, ax = plt.subplots()
    # plot_tran_data(ax, "spice_simulation_raw/nmem_cell_write_1000uA.raw", "I(ichr)")
    # plt.show()
    # fig, ax = plt.subplots()
    # plot_tran_data(ax, "spice_simulation_raw/nmem_cell_write_step_5uA.raw", "I(ichr)")
    # plt.show()
    # fig, ax = plt.subplots()
    # plot_tran_data(ax, "spice_simulation_raw/nmem_cell_write_step_10uA.raw", "I(ichr)")
    # plt.show()


def example_step_enable_write_sweep_write() -> None:
    data_dict = import_raw_dir("spice_simulation_raw/enable_write_sweep/")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    colors = CMAP(np.linspace(0, 1, len(data_dict)))
    for i, (key, l) in enumerate(data_dict.items()):
        processed_data = process_read_data(l)
        write_current = key[-9:-6]
        ax = axs[0]
        ax = plot_current_sweep_ber(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )
        ax.set_ylabel("BER")

        ax = axs[1]
        ax = plot_current_sweep_output(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )
        ax.set_ylabel("Output Voltage (mV)")

        ax = axs[2]
        ax = plot_current_sweep_persistent(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )

    for ax in axs:

        ax.legend(loc="upper left", ncol=1, prop={"size": 12}, bbox_to_anchor=(1.05, 1))
    plt.show()


if __name__ == "__main__":
    # example_write_read_clear()
    example_step_read_sweep_write()
    # example_step_enable_write_sweep_write()
