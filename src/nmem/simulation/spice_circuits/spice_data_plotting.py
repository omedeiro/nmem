import ltspice
import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import CMAP
from nmem.simulation.spice_circuits.plotting import (
    plot_tran_data,
    plot_current_sweep_ber,
)
from nmem.simulation.spice_circuits.functions import (
    get_write_sweep_data,
    import_raw_dir,
    process_data_dict_sweep,
    process_read_data,
    get_step_parameter,
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

    # for i, (key, data) in enumerate(processed_data.items()):
    #     step_parameter, zero_currents, one_currents, step_parameter_str = process_data_dict_sweep(
    #         data
    #     )
    #     fig, ax = plt.subplots()
    #     ax.plot(step_parameter, zero_currents, "-o", label="Read 0")
    #     ax.plot(step_parameter, one_currents, "-o", label="Read 1")
    #     ax.set_ylabel("Read Current (uA)")
    #     ax.set_xlabel(f"{step_parameter_str}")
    #     read_margin = zero_currents - one_currents
    #     optimal_idx = np.argmax(read_margin)

    #     ax.plot(
    #         step_parameter[optimal_idx],
    #         zero_currents[optimal_idx],
    #         "x",
    #         label="Optimal Read Margin",
    #     )
    #     optimal_read = (
    #         np.array(zero_currents[optimal_idx]) - np.array(read_margin[optimal_idx]) / 2
    #     )
    #     ax.plot(
    #         step_parameter[optimal_idx],
    #         optimal_read,
    #         "x",
    #         label="Optimal Read Margin",
    #     )
    #     ax.legend()
    #     ax.title = f"Write {write_current}uA"
    #     plt.show()


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
    fig, ax = plt.subplots()
    colors = CMAP(np.linspace(0, 1, len(data_dict)))
    for i, (key, l) in enumerate(data_dict.items()):
        processed_data = process_read_data(l)
        write_current = key[-9:-6]
        ax = plot_current_sweep_ber(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )
    ax.legend(loc="upper left", ncol=1, prop={"size": 12}, bbox_to_anchor=(1.05, 1))
    plt.show()


if __name__ == "__main__":
    # example_write_read_clear()
    # example_step_read_sweep_write()
    example_step_enable_write_sweep_write()
