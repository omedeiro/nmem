import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_fill_between_array,
    plot_read_sweep_array,
)

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams.update(
    {
        "figure.figsize": [3.5, 3.5],
        "font.size": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Inter",
        "lines.markersize": 2,
        "lines.linewidth": 1.2,
        "legend.fontsize": 5,
        "legend.frameon": False,
        "xtick.major.size": 1,
        "ytick.major.size": 1,
    }
)

if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]

    fig, axs = plt.subplots(1, 3, figsize=(7, 4.3), sharey=True)
    for i in range(3):
        plot_read_sweep_array(
            axs[i], dict_list[i], "bit_error_rate", "enable_read_current"
        )
        plot_fill_between_array(axs[i], dict_list[i])
        axs[i].set_xlim(400, 1000)
        axs[i].set_ylim(0, 1)
        axs[i].set_xlabel("Read Current ($\mu$A)")

    axs[0].set_ylabel("Bit Error Rate")
    axs[2].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Read Current,\n Read Temperature",
    )

    plt.savefig("read_current_sweep_three.pdf", bbox_inches="tight")
