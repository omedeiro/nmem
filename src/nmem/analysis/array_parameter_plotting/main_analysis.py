import matplotlib.pyplot as plt
from matplotlib import font_manager
from nmem.analysis.analysis import (
    convert_cell_to_coordinates,
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
    plot_array(
        axs["bit_error"],
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )
    axs["bit_error"].set_title("Bit Error Rate")
    plot_array(
        axs["write"],
        xloc_list,
        yloc_list,
        param_dict["write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    axs["write"].set_title("Write Current (uA)")
    plot_array(
        axs["read"],
        xloc_list,
        yloc_list,
        param_dict["read_current"],
        log=False,
        cmap=plt.get_cmap("Blues"),
    )
    axs["read"].set_title("Read Current (uA)")
    plot_array(
        axs["enable_write"],
        xloc_list,
        yloc_list,
        param_dict["enable_write_current"],
        log=False,
        cmap=plt.get_cmap("Reds"),
    )
    axs["enable_write"].set_title("Enable Write Current (uA)")
    plot_array(
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
