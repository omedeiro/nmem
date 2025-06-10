import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    initialize_dict,
    process_cell,
)
from nmem.analysis.plotting import plot_parameter_array
from nmem.analysis.utils import convert_cell_to_coordinates
from nmem.measurement.cells import CELLS


def main(array_size=(4, 4), save=False, save_path="main_analysis.pdf"):
    """
    Generate and plot parameter arrays for the memory cell array.

    Parameters
    ----------
    array_size : tuple
        Size of the array (default (4, 4)).
    save : bool
        If True, save the figure to save_path.
    save_path : str
        Path to save the figure if save is True.
    """
    param_dict = initialize_dict(array_size)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    fig, axs = plt.subplot_mosaic(
        [
            ["bit_error", "write", "read"],
            ["bit_error", "enable_write", "enable_read"],
        ],
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
    if save:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
