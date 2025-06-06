
import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_enable_sweep,
    plot_enable_sweep_markers,
    set_plot_style,
)

set_plot_style()
C0 = "#1b9e77"
C1 = "#d95f02"
RBCOLORS = plt.get_cmap("coolwarm")(np.linspace(0, 1, 4))
CMAP2 = plt.get_cmap("viridis")


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
        inner,
        figsize=(6, 2),
    )

    dict_list = import_directory("data")
    sort_dict_list = sorted(
        dict_list, key=lambda x: x.get("write_current").flatten()[0]
    )

    ax = axs["A"]
    plot_enable_sweep(
        ax, sort_dict_list, range=slice(0, len(sort_dict_list), 2), add_colorbar=True
    )

    ax = axs["B"]
    plot_enable_sweep_markers(ax, sort_dict_list)

    axpos = axs["A"].get_position()
    ax2pos = axs["B"].get_position()
    fig.subplots_adjust(wspace=0.7, hspace=0.5)
