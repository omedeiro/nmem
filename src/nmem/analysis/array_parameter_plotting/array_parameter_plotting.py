import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    initialize_dict,
    process_cell,
)
from nmem.analysis.plotting import plot_parameter_array
from nmem.analysis.utils import convert_cell_to_coordinates
from nmem.measurement.cells import CELLS


def main(array_size=(4, 4)):
    """
    Process cell data and plot parameter arrays for the given array size.
    """
    xloc_list = []
    yloc_list = []
    param_dict = initialize_dict(array_size)
    yintercept_list = []
    slope_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)
        yintercept = CELLS[c]["y_intercept"]
        yintercept_list.append(yintercept)
        slope = CELLS[c]["slope"]
        slope_list.append(slope)
    fig, ax = plt.subplots()
    plot_parameter_array(
        ax, xloc_list, yloc_list, param_dict["write_current"], "Write Current [$\mu$A]"
    )
    plt.show()


if __name__ == "__main__":
    main()
