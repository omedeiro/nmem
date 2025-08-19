import os

import ltspice
import numpy as np
from matplotlib import pyplot as plt

from nmem.analysis.utils import filter_first
from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import CMAP
from nmem.analysis.state_currents_plots import (
    plot_current_sweep_ber,
    plot_current_sweep_switching,
    plot_case,
    plot_case_vout,
)
from nmem.analysis.sweep_plots import (
    plot_read_sweep_array,
    plot_read_switch_probability_array,
)

VOLTAGE_THRESHOLD = 2.0e-3

def get_ltsp_ber(
    read_zero_voltage: float,
    read_one_voltage: float,
    voltage_threshold: float = VOLTAGE_THRESHOLD,
) -> float:
    ber = np.where(
        (read_one_voltage < voltage_threshold)
        & (read_zero_voltage > voltage_threshold),
        1,
        0.5,
    )
    ber = np.where(
        (read_one_voltage > voltage_threshold)
        & (read_zero_voltage < voltage_threshold),
        0,
        ber,
    )
    return ber

def get_ltsp_prob(
    read_zero_voltage: float,
    read_one_voltage: float,
    voltage_threshold: float = VOLTAGE_THRESHOLD,
) -> float:
    switching_probability = np.where(
        (read_one_voltage > voltage_threshold)
        & (read_zero_voltage > voltage_threshold),
        1,
        0.5,
    )
    switching_probability = np.where(
        (read_one_voltage < voltage_threshold)
        & (read_zero_voltage < voltage_threshold),
        0,
        switching_probability,
    )
    return switching_probability



def safe_max(arr: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return np.max(arr[mask])
    return 0


def safe_min(arr: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return np.min(arr[mask])
    return 0

def get_current_or_voltage(
    ltsp: ltspice.Ltspice, signal: str, case: int = 0
) -> np.ndarray:
    signal_data = ltsp.get_data(f"I({signal})", case=case)
    if signal_data is None:
        signal_data = ltsp.get_data(f"V({signal})", case=case)
    return signal_data * 1e6

def process_read_data(ltsp: ltspice.Ltspice) -> dict:
    num_cases = ltsp.case_count

    read_current = np.zeros(num_cases)
    enable_read_current = np.zeros(num_cases)
    enable_write_current = np.zeros(num_cases)
    write_current = np.zeros(num_cases)
    persistent_current = np.zeros(num_cases)
    write_one_voltage = np.zeros(num_cases)
    write_zero_voltage = np.zeros(num_cases)
    read_zero_voltage = np.zeros(num_cases)
    read_one_voltage = np.zeros(num_cases)
    read_margin = np.zeros(num_cases)
    bit_error_rate = np.zeros(num_cases)
    switching_probability = np.zeros(num_cases)

    time_windows = {
        "persistent_current": (1.5e-7, 2e-7),
        "write_one": (1e-7, 1.5e-7),
        "write_zero": (5e-7, 5.5e-7),
        "read_one": (2e-7, 2.5e-7),
        "read_zero": (4e-7, 4.5e-7),
        "enable_write": (1e-7, 1.5e-7),
    }
    data_dict = {}
    for i in range(num_cases):
        time = ltsp.get_time(i)

        enable_current = ltsp.get_data("I(R1)", i) * 1e6
        channel_current = ltsp.get_data("I(R2)", i) * 1e6
        right_branch_current = ltsp.get_data("Ix(HR:drain)", i) * 1e6
        left_branch_current = ltsp.get_data("Ix(HL:drain)", i) * 1e6
        left_critical_current = get_current_or_voltage(ltsp, "ichl", i)
        right_critical_current = get_current_or_voltage(ltsp, "ichr", i)
        left_retrapping_current = get_current_or_voltage(ltsp, "irhl", i)
        right_retrapping_current = get_current_or_voltage(ltsp, "irhr", i)
        output_voltage = ltsp.get_data("V(out)", i)
        masks = {
            key: (time > start) & (time < end)
            for key, (start, end) in time_windows.items()
        }

        persistent_current[i] = safe_max(
            right_branch_current, masks["persistent_current"]
        )
        write_current[i] = safe_max(channel_current, masks["write_one"])
        read_current[i] = safe_max(channel_current, masks["read_one"])
        enable_read_current[i] = safe_max(enable_current, masks["read_one"])
        enable_write_current[i] = safe_max(enable_current, masks["enable_write"])
        write_one_voltage[i] = safe_max(output_voltage, masks["write_one"])
        write_zero_voltage[i] = safe_min(output_voltage, masks["write_zero"])
        read_zero_voltage[i] = safe_max(output_voltage, masks["read_zero"])
        read_one_voltage[i] = safe_max(output_voltage, masks["read_one"])
        read_margin[i] = read_zero_voltage[i] - read_one_voltage[i]
        bit_error_rate[i] = get_ltsp_ber(
            read_zero_voltage[i], read_one_voltage[i]
        )
        switching_probability[i] = get_ltsp_prob(
            read_zero_voltage[i], read_one_voltage[i]
        )
        data_dict[i] = {
            "time": time,
            "tran_enable_current": enable_current,
            "tran_channel_current": channel_current,
            "tran_right_branch_current": right_branch_current,
            "tran_left_branch_current": left_branch_current,
            "tran_left_critical_current": left_critical_current,
            "tran_right_critical_current": right_critical_current,
            "tran_left_retrapping_current": left_retrapping_current,
            "tran_right_retrapping_current": right_retrapping_current,
            "tran_output_voltage": output_voltage,
            "write_current": write_current,
            "read_current": read_current,
            "enable_write_current": enable_write_current,
            "enable_read_current": enable_read_current,
            "read_zero_voltage": read_zero_voltage,
            "read_one_voltage": read_one_voltage,
            "write_one_voltage": write_one_voltage,
            "write_zero_voltage": write_zero_voltage,
            "persistent_current": persistent_current,
            "case_count": ltsp.case_count,
            "read_margin": read_margin,
            "bit_error_rate": bit_error_rate,
            "switching_probability": switching_probability,
        }
    return data_dict




def create_plot_stable(
    axs: list[plt.Axes], data_dict: dict, cases: list[int]
) -> list[plt.Axes]:

    write_current = data_dict[0]["write_current"][0]

    time_windows = {
        0: (100e-9, 150e-9),
        1: (200e-9, 250e-9),
        2: (300e-9, 350e-9),
        3: (400e-9, 450e-9),
    }
    sweep_param_list = []
    for case in cases:
        for i, time_window in time_windows.items():
            sweep_param = data_dict[case]["read_current"]
            print(f"sweep param {sweep_param}")
            sweep_param = sweep_param[case]
            sweep_param_list.append(sweep_param)
            ax: plt.Axes = axs[f"T{i}"]
            plot_case(ax, data_dict, case, "left")
            plot_case(ax, data_dict, case, "right")
            ax.plot(
                data_dict[case]["time"],
                -1 * data_dict[case]["tran_left_critical_current"],
                color="C0",
                linestyle="--",
            )
            ax.plot(
                data_dict[case]["time"],
                -1 * data_dict[case]["tran_right_critical_current"],
                color="C1",
                linestyle="--",
            )
            ax.plot(
                data_dict[case]["time"],
                data_dict[case]["tran_left_retrapping_current"],
                color="C4",
                linestyle=":",
            )
            ax.plot(
                data_dict[case]["time"],
                data_dict[case]["tran_right_retrapping_current"],
                color="C5",
                linestyle=":",
            )
            ax.set_ylim(-300, 900)
            ax.set_xlim(time_window)
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))

            ax.set_ylabel("I ($\mu$A)", labelpad=-4)
            ax.set_xlabel("Time (ns)", labelpad=-3)
            ax.yaxis.set_major_locator(plt.MultipleLocator(250))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(50))
            ax.xaxis.set_major_locator(plt.MultipleLocator(50e-9))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10e-9))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e9:.0f}"))



            ax: plt.Axes = axs[f"B{i}"]
            plot_case_vout(ax, data_dict, case, "tran_output_voltage", color="k")
            ax.set_ylim(-50e-3, 50e-3)
            ax.set_xlim(time_window)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.yaxis.set_major_locator(plt.MultipleLocator(50e-3))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.0f}"))
            ax.set_xlabel("Time (ns)", labelpad=-3)
            ax.set_ylabel("V (mV)", labelpad=-3)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50e-9))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10e-9))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e9:.0f}"))
    return axs



def main():

    # Get and parse raw files
    files = [
        f for f in os.listdir("../data/ber_sweep_read_current/ltspice_simulation/") if f.endswith(".raw")
    ]
    parsed_data = {}
    write_current_list = []

    for file in files:
        data = ltspice.Ltspice(f"../data/ber_sweep_read_current/ltspice_simulation/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        parsed_data[file] = ltsp_data_dict
        write_current = ltsp_data_dict[0]["write_current"][0] * 1e6
        write_current_list.append(write_current)

    # Sort files and data by write current
    sorted_files_data = sorted(zip(files, write_current_list), key=lambda x: x[1])
    files, write_current_list = zip(*sorted_files_data)

    # Parse example trace once
    ltsp_data_dict = parsed_data["nmem_cell_read_example_trace.raw"]

    inner = [
        ["T0", "T1", "T2", "T3"],
    ]
    innerb = [
        ["B0", "B1", "B2", "B3"],
    ]
    inner2 = [
        ["A", "B"],
    ]
    inner3 = [
        ["C", "D"],
    ]
    outer_nested_mosaic = [
        [inner],
        [innerb],
        [inner2],
        [inner3],
    ]
    fig, axs = plt.subplot_mosaic(
        outer_nested_mosaic,
        figsize=(180 / 25.4, 180 / 25.4),
        height_ratios=[2, 0.5, 1, 1],
    )

    CASE = 16
    create_plot_stable(axs, ltsp_data_dict, cases=[CASE])
    print(ltsp_data_dict[0]["read_current"])
    case_current = ltsp_data_dict[CASE]["read_current"][CASE]

    handles, labels = axs["T0"].get_legend_handles_labels()
    # Select specific items
    selected_labels = [
        "Left Branch Current",
        "Right Branch Current",
        "Left Critical Current",
        "Right Critical Current",
    ]
    selected_labels2 = [
        "$i_{\mathrm{H_L}}$",
        "$i_{\mathrm{H_R}}$",
        "$I_{\mathrm{c,H_L}}$",
        "$I_{\mathrm{c,H_R}}$",
    ]
    selected_handles = [handles[labels.index(lbl)] for lbl in selected_labels]

    dict_list = import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_C3")[::2]
    write_current_list = [
        filter_first(data_dict["write_current"]) * 1e6 for data_dict in dict_list
    ]

    # Sort dict_list and write_current_list together
    sorted_dicts = sorted(zip(dict_list, write_current_list), key=lambda x: x[1])
    dict_list, write_current_list = zip(*sorted_dicts)

    plot_read_sweep_array(
        axs["A"],
        dict_list,
        "bit_error_rate",
        "write_current",
        marker=".",
        linestyle="-",
        markersize=4,
    )
    axs["A"].set_xlim(650, 850)
    axs["A"].set_ylabel("BER")
    axs["A"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    plot_read_switch_probability_array(
        axs["B"], dict_list, write_current_list, marker=".", linestyle="-", markersize=2
    )
    axs["B"].set_xlim(650, 850)
    axs["B"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    axs["D"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)

    axs["C"].set_xlim(650, 850)
    axs["D"].set_xlim(650, 850)
    axs["C"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    axs["C"].set_ylabel("BER")
    axs["B"].set_ylabel("Switching Probability")
    axs["D"].set_ylabel("Switching Probability")

    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    col_set = [colors[i] for i in [0, 2, -1]]
    selected_files = [files[i] for i in [0, 2, 11]]
    max_write_current = 300

    for file in selected_files:
        ltsp_data_dict = parsed_data[file]
        ltsp_write_current = ltsp_data_dict[0]["write_current"][0]
        normalized_color = CMAP(ltsp_write_current / max_write_current)

        plot_current_sweep_ber(
            axs["C"],
            ltsp_data_dict,
            color=normalized_color,
            label=f"{ltsp_write_current} $\mu$A",
            marker=".",
            linestyle="-",
            markersize=5,
        )
        plot_current_sweep_switching(
            axs["D"],
            ltsp_data_dict,
            color=normalized_color,
            label=f"{ltsp_write_current} $\mu$A",
            marker=".",
            markersize=5,
        )

    axs["A"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["B"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["C"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["D"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)

    axs["B"].legend(
        loc="upper right",
        labelspacing=0.1,
        fontsize=6,
    )
    axs["D"].legend(
        loc="upper right",
        labelspacing=0.1,
        fontsize=6,
    )

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_alpha(0)

    ax_legend = fig.add_axes([0.5, 0.9, 0.1, 0.01])
    ax_legend.axis("off")
    ax_legend.legend(
        selected_handles,
        selected_labels2,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        handlelength=2.5,
        fontsize=8,
    )

    plt.savefig("../results/figure3.pdf")


if __name__ == "__main__":
    main()