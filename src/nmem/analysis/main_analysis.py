import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from nmem.analysis.analysis import (
    convert_location_to_coordinates,
    initialize_dict,
    plot_array,
    process_cell,
)
from nmem.measurement.cells import CELLS

# font_path = "/home/omedeiro/Inter-Regular.otf"
font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [7, 3.5]
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.labelpad"] = 0.5


def plot_array_3d(
    xloc, yloc, ztotal, title=None, log=False, norm=False, reverse=False, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    cmap = plt.cm.get_cmap("viridis")
    if reverse:
        cmap = plt.cm.get_cmap("viridis").reversed()

    ax.bar3d(xloc, yloc, 0, 1, 1, ztotal.flatten(), shade=True)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.set_zlim(0, np.nanmax(ztotal))
    ax.patch.set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=6, pad=0)
    # ax = plot_text_labels(xloc, yloc, ztotal, log, ax=ax)

    return ax


if __name__ == "__main__":
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_location_to_coordinates(c)
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
    plot_array(
        axs["bit_error"],
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )

    plot_array(
        axs["write"],
        xloc_list,
        yloc_list,
        param_dict["write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    plot_array(
        axs["read"],
        xloc_list,
        yloc_list,
        param_dict["read_current"],
        log=False,
        cmap=plt.get_cmap("Blues"),
    )
    plot_array(
        axs["enable_write"],
        xloc_list,
        yloc_list,
        param_dict["enable_write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    plot_array(
        axs["enable_read"],
        xloc_list,
        yloc_list,
        param_dict["enable_read_current"],
        log=False,
        cmap=plt.get_cmap("Blues"),
    )

    fig.patch.set_visible(False)
    # plt.savefig("main_analysis.pdf", bbox_inches="tight")
