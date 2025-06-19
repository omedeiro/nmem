import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    process_array_parameter_data,  # newly refactored function
)
from nmem.analysis.matrix_plots import plot_parameter_array
from nmem.measurement.cells import CELLS


def main(save_dir=None):
    """
    Process cell data and plot parameter arrays for the given array size.
    """
    xloc_list, yloc_list, param_dict, yintercept_list, slope_list = (
        process_array_parameter_data(CELLS)
    )
    fig, ax = plt.subplots()
    plot_parameter_array(
        xloc_list,
        yloc_list,
        param_dict["write_current"],
        "Write Current [$\\mu$A]",
        ax=ax,
        save_path=f"{save_dir}/write_current_array.png" if save_dir else None,
    )


if __name__ == "__main__":
    main(save_dir="../plots")
