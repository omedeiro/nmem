import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    convert_cell_to_coordinates,
    initialize_dict,
    process_cell,
)
from nmem.analysis.plotting import plot_parameter_array
from nmem.measurement.cells import CELLS

if __name__ == "__main__":
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    fig, axs = plt.subplot_mosaic(
        [
            [
                "bit_error",
                "write",
                "read",
            ],
            [
                "bit_error",
                "enable_write",
                "enable_read",
            ],
        ],
        # per_subplot_kw={"bit_error": {"projection": "3d"}},
    )
    plot_parameter_array(
        axs["bit_error"],
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )
    axs["bit_error"].set_title("Bit Error Rate")
    plot_parameter_array(
        axs["write"],
        xloc_list,
        yloc_list,
        param_dict["write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    axs["write"].set_title("Write Current (uA)")
    plot_parameter_array(
        axs["read"],
        xloc_list,
        yloc_list,
        param_dict["read_current"],
        log=False,
        cmap=plt.get_cmap("Blues"),
    )
    axs["read"].set_title("Read Current (uA)")
    plot_parameter_array(
        axs["enable_write"],
        xloc_list,
        yloc_list,
        param_dict["enable_write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    axs["enable_write"].set_title("Enable Write Current (uA)")
    plot_parameter_array(
        axs["enable_read"],
        xloc_list,
        yloc_list,
        param_dict["enable_read_current"],
        log=False,
        cmap=plt.get_cmap("Blues"),
    )
    axs["enable_read"].set_title("Enable Read Current (uA)")

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_visible(False)
    save = False
    if save:
        plt.savefig("main_analysis.pdf", bbox_inches="tight")
