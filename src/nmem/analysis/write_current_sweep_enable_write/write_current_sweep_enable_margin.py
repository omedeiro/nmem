import os

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.data_import import (
    import_write_sweep_formatted,
    import_write_sweep_formatted_markers,
)
from nmem.analysis.plotting import (
    set_plot_style,
    plot_write_sweep_formatted,
    plot_write_sweep_formatted_markers,
)
from nmem.analysis.data_import import import_directory
set_plot_style()

C0 = "#1b9e77"
C1 = "#d95f02"
RBCOLORS = plt.get_cmap("coolwarm")(np.linspace(0, 1, 4))




if __name__ == "__main__":
    inner = [
        ["A", "B"],
    ]
    innerb = [
        ["C", "D"],
    ]
    innerc = [
        ["delay", "bergrid"],
    ]
    outer_nested_mosaic = [
        [inner],
        [innerb],
        [innerc],
    ]

    fig, axs = plt.subplot_mosaic(
        innerb,
        figsize=(6, 2),
    )

    dict_list = import_write_sweep_formatted()
    plot_write_sweep_formatted(axs["C"], dict_list)

    data_dict = import_write_sweep_formatted_markers(dict_list)
    plot_write_sweep_formatted_markers(axs["D"], data_dict)


    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.1,
        hspace=0.4,
        wspace=0.7,
    )
