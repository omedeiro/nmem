import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from nmem.analysis.analysis import (
    convert_cell_to_coordinates,
    get_bit_error_rate,
    import_directory,
    initialize_dict,
    plot_parameter_array,
    process_cell,
)
from nmem.measurement.cells import CELLS

if __name__ == "__main__":
    dict_list = import_directory("data3")
    # dict_list.extend(import_directory("data3"))
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()[0] * 1e-3
        bit_error_rate = get_bit_error_rate(data_dict)

        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)

        # print(
        #     f"delay: {delay}, bit_error_rate: {bit_error_rate}, num_measurements: {data_dict['num_meas'].flatten()[0]:.0g}"
        # )
    fidelity = 1 - np.array(bit_error_rate_list)

    # fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    fig, axs = plt.subplot_mosaic("A;B", figsize=(3.5, 3.5), constrained_layout=True)
    ax = axs["A"]
    ax.set_aspect("equal")
    sort_index = np.argsort(delay_list)
    delay_list = np.array(delay_list)[sort_index]
    bit_error_rate_list = np.array(bit_error_rate_list)[sort_index]
    ax.plot(delay_list, bit_error_rate_list, marker="o", color="black")
    ax.set_ylabel("BER")
    ax.set_xlabel("Memory Retention Time (s)")

    ax.set_xscale("log")
    ax.set_xbound(lower=1e-6)
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    ax.grid(True, which="both", linestyle="--")

    ax.set_yscale("log")
    ax.set_ylim([1e-4, 1e-3])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # plt.savefig("read_delay_retention_test.png", dpi=300, bbox_inches="tight")
    # plt.show()
    ax = axs["B"]
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    plot_parameter_array(
        ax,
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )
    cax = ax.inset_axes([1.10, 0, 0.1, 1])
    cbar = fig.colorbar(
        ax.get_children()[0], cax=cax, orientation="vertical", label="minimum BER"
    )
    # cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2])

    
    plt.savefig("read_delay_retention_test.pdf", bbox_inches="tight")
    plt.show()
