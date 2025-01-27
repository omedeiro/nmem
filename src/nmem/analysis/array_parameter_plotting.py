import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    convert_location_to_coordinates,
    initialize_dict,
    plot_array,
    process_cell,
)
from nmem.measurement.cells import CELLS

# plt.rcParams["figure.figsize"] = [10, 12]
plt.rcParams["font.size"] = 10


if __name__ == "__main__":
    xloc_list = []
    yloc_list = []
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    for c in CELLS:
        xloc, yloc = convert_location_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    fig, ax = plt.subplots()
    plot_array(
        ax, xloc_list, yloc_list, param_dict["write_current"], "Write Current [$\mu$A]"
    )
