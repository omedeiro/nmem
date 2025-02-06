import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
    get_write_temperature,
    get_write_temperatures,
    get_read_temperatures_array,
    get_enable_read_currents_array,
)


font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 6
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False

plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1


if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]

    # fig, axs = plt.subplots(1, 3, figsize=(7, 4.3), sharey=True)
    # for i in range(3):
    #     plot_read_sweep_array(
    #         axs[i], dict_list[i], "bit_error_rate", "enable_read_current"
    #     )
    #     axs[i].set_xlim(400, 1000)
    #     axs[i].set_ylim(0, 1)
    #     axs[i].set_xlabel("Read Current ($\mu$A)")
    #     enable_write_temp = get_write_temperature(dict_list[i][0])
    #     print(f"Enable Write Temp: {enable_write_temp}")
    # axs[0].set_ylabel("Bit Error Rate")
    # axs[2].legend(
    #     frameon=False,
    #     loc="upper left",
    #     bbox_to_anchor=(1, 1),
    #     title="Enable Read Current,\n Read Temperature",
    # )

    # plt.savefig("read_current_sweep_enable_read.png", dpi=300, bbox_inches="tight")

    fig, axs = plt.subplots(1,2, figsize=(7.5, 2), constrained_layout=True, width_ratios=[1, .25])

    dict_list = dict_list[2]
    ax = axs[0]
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_read_current")
    ax.axvline(910, color="black", linestyle="--")
    ax.set_xlabel("$I_{\mathrm{read}}$ ($\mu$A)")
    ax.set_ylabel("BER")

    read_temperatures = get_read_temperatures_array(dict_list)
    enable_read_currents = get_enable_read_currents_array(dict_list)

    ax.set_xlim(400, 1000)
    ax = axs[1]
    ax.plot(
        enable_read_currents,
        read_temperatures,
        marker="o",
        color="black",
        linestyle="--",
    )
    ax.set_xlabel("$I_{\mathrm{enable}}$ ($\mu$A)")
    ax.set_ylabel("$T_{\mathrm{read}}$ (K)")
    ax.yaxis.set_major_locator(plt.MultipleLocator(.2))
    plt.savefig("read_current_sweep_enable_read2.pdf", bbox_inches="tight")