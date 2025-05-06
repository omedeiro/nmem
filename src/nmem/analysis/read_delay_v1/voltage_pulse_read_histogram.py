import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nmem.analysis.analysis import (
    get_bit_error_rate,
    import_directory,
    plot_voltage_hist,
    plot_voltage_trace_averaged,
    set_inter_font,
    set_pres_style,
)

set_inter_font()
set_pres_style()
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
})
# # plt.rcParams["figure.figsize"] = [7, 3.5]
# plt.rcParams["font.size"] = 5
# plt.rcParams["axes.linewidth"] = 0.5
# plt.rcParams["xtick.major.width"] = 0.5
# plt.rcParams["ytick.major.width"] = 0.5
# # plt.rcParams["xtick.direction"] = "in"
# # plt.rcParams["ytick.direction"] = "in"
# # plt.rcParams["font.family"] = "Inter"
# # plt.rcParams["lines.markersize"] = 2
# plt.rcParams["lines.linewidth"] = 0.5
# plt.rcParams["legend.fontsize"] = 5
# plt.rcParams["legend.frameon"] = False


# plt.rcParams["xtick.major.size"] = 1
# plt.rcParams["ytick.major.size"] = 1


def plot_read_delay(ax: Axes, dict_list: list[dict]) -> Axes:
    bers = []
    for i in range(4):
        bers.append(get_bit_error_rate(dict_list[i]))

    ax.plot([1, 2, 3, 4], bers, label="bit_error_rate", marker="o", color="#345F90")
    ax.set_xlabel("Delay [µs]")
    ax.set_ylabel("BER")

    return ax


def create_trace_hist_plot(
    ax_dict: dict[str, Axes], dict_list: list[dict], save: bool = False
) -> Axes:
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()

    plot_voltage_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_voltage_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff7f0e",
        linestyle="--",
        label="Read 1",
    )
    plot_voltage_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )

    plot_voltage_hist(ax_dict["C"], dict_list[-1])

    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [µs]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    return ax_dict

def compute_sigma_separation(data: dict, show_print=True) -> float:
    """Compute the peak separation between read0 and read1 histograms in units of σ."""
    v_read0 = np.array(data["read_zero_top"])
    v_read1 = np.array(data["read_one_top"])
    
    # Remove NaNs or invalid data
    v_read0 = v_read0[np.isfinite(v_read0)]
    v_read1 = v_read1[np.isfinite(v_read1)]
    
    mu0 = np.mean(v_read0)
    mu1 = np.mean(v_read1)
    sigma0 = np.std(v_read0)
    sigma1 = np.std(v_read1)

    sigma_avg = 0.5 * (sigma0 + sigma1)
    separation_sigma = np.abs(mu1 - mu0) / sigma_avg

    if show_print:
        print(f"μ0 = {mu0:.3f} mV, σ0 = {sigma0:.3f} mV")
        print(f"μ1 = {mu1:.3f} mV, σ1 = {sigma1:.3f} mV")
        print(f"Separation = {separation_sigma:.2f} σ")

    return separation_sigma


# if __name__ == "__main__":
#     dict_list = import_directory("data")

#     # fig, ax_dict = plt.subplot_mosaic("AC;BC", figsize=(6, 2.5), constrained_layout=True)
#     fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
#     # ax2 = ax_dict["A"].twinx()
#     # ax3 = ax_dict["B"].twinx()

#     # plot_voltage_trace_averaged(
#     #     ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
#     # )
#     # plot_voltage_trace_averaged(
#     #     ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable\nWrite"
#     # )
#     # plot_voltage_trace_averaged(
#     #     ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
#     # )
#     # plot_voltage_trace_averaged(
#     #     ax_dict["B"],
#     #     dict_list[4],
#     #     "trace_read1_avg",
#     #     color="#ff7f0e",
#     #     linestyle="--",
#     #     label="Read 1",
#     # )
#     # plot_voltage_trace_averaged(
#     #     ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable\nRead"
#     # )

#     plot_voltage_hist(ax, dict_list[3])
#     sigma_sep = compute_sigma_separation(dict_list[3], show_print=True)
#     # ax_dict["A"].legend(loc="upper left", handlelength=1.2)
#     # ax_dict["A"].set_ylabel("$V$ [mV]")
#     # ax2.legend(loc="upper right", handlelength=1.2)
#     # ax2.set_ylabel("$V$ [mV]")
#     # ax3.legend(loc="upper right", handlelength=1.2)
#     # ax3.set_ylabel("$V$ [mV]")
#     # ax_dict["B"].set_xlabel("$t$ [µs]")
#     # ax_dict["B"].set_ylabel("$V$ [mV]")
#     # ax_dict["B"].legend(loc="upper left", handlelength=1.2)
#     ax.set_xlabel("Voltage [mV]")
#     ax.set_ylabel("Counts")
#     # fig.subplots_adjust(wspace=0.5, hspace=0.5)
#     plt.savefig("delay_plotting_hist.png", bbox_inches="tight")


if __name__ == "__main__":
    dict_list = import_directory("data")
    data = dict_list[3]

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    # Plot the histogram
    plot_voltage_hist(ax, data)

      # Extract data and compute stats
    v_read0 = np.array(data["read_zero_top"])
    v_read1 = np.array(data["read_one_top"])
    v_read0 = v_read0[np.isfinite(v_read0)]
    v_read1 = v_read1[np.isfinite(v_read1)]

    # 🔁 Convert to millivolts
    v_read0 *= 1e3
    v_read1 *= 1e3

    mu0, sigma0 = np.mean(v_read0), np.std(v_read0)
    mu1, sigma1 = np.mean(v_read1), np.std(v_read1)
    sigma_avg = 0.5 * (sigma0 + sigma1)
    separation_sigma = np.abs(mu1 - mu0) / sigma_avg

    # ➕ Compute 3σ bound spacing
    bound_diff_3sigma = (mu1 - 4*sigma1) - (mu0 + 4*sigma0)

    print(f"μ0 = {mu0:.3f} mV, σ0 = {sigma0:.3f} mV")
    print(f"μ1 = {mu1:.3f} mV, σ1 = {sigma1:.3f} mV")
    print(f"Separation = {separation_sigma:.2f} σ")
    print(f"3σ Bound Difference = {bound_diff_3sigma:.2f} mV")

    # Plot vertical lines at 3σ bounds
    ax.axvline(mu0 + 4*sigma0, color="gray", linestyle="--", label="_μ₀ - 3σ₀")
    ax.axvline(mu1 - 4*sigma1, color="gray", linestyle="--", label="_μ₁ + 3σ₁")

    # Annotate σ-separation and 3σ span
    # textstr = f"Separation = {separation_sigma:.2f} σ\n3σ bound diff = {bound_diff_3sigma:.1f} mV"
    # props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)
    # ax.text(
    #     0.98, 0.95, textstr,
    #     transform=ax.transAxes,
    #     fontsize=12,
    #     verticalalignment='top',
    #     horizontalalignment='right',
    #     bbox=props
    # )


    ax.set_xlabel("Voltage [mV]")
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    plt.savefig("delay_plotting_hist.png", bbox_inches="tight")
