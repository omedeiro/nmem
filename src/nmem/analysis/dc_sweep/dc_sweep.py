import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from nmem.analysis.analysis import (
    import_directory,
    plot_critical_currents_from_dc_sweep,
    plot_current_voltage_from_dc_sweep,
)

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 3.5]
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


plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1


def plot_combined_figure(ax: Axes, dict_list: list, save: bool = False) -> Axes:
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 0].axis("off")
    ax[1, 1] = plot_current_voltage_from_dc_sweep(ax[1, 1], dict_list)
    ax[1, 2] = plot_critical_currents_from_dc_sweep(ax[1, 2], dict_list)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("iv_curve_combined.pdf", bbox_inches="tight")

    return ax


if __name__ == "__main__":
    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_critical_currents_from_dc_sweep(ax, data_list)
    plt.show()

    fig, ax = plt.subplots()
    plot_current_voltage_from_dc_sweep(ax, data_list)
    plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(7, 3.5), height_ratios=[1, 0.7])
    plot_combined_figure(ax, data_list)
    plt.show()
