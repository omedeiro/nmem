from typing import Literal

import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm
from nmem.simulation.spice_circuits.functions import (
    get_step_parameter,
    process_read_data,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.io as sio
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
    plot_read_switch_probability_array,
    filter_first,
)

CMAP = plt.get_cmap("coolwarm")


FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3


def set_inter_font():
    if os.name == "nt":  # Windows
        font_path = r"C:\Users\ICE\AppData\Local\Microsoft\Windows\Fonts\Inter-VariableFont_opsz,wght.ttf"
    elif os.name == "posix":
        font_path = "/home/omedeiro/Inter-VariableFont_opsz,wght.ttf"
    else:
        font_path = None

    if font_path and os.path.exists(font_path):
        print(f"Font path: {font_path}")
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "Inter"



def apply_snm_style():
    mpl.rcParams.update({
        "figure.figsize": [3.5, 3.5],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.markersize": 2,
        "lines.linewidth": 1.5,
        "legend.fontsize": 5,
        "legend.frameon": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
    })


# COLORS = [
#     "#1b9e77",  # Teal green
#     "#d95f02",  # Burnt orange
#     "#7570b3",  # Muted blue-purple
#     "#e7298a",  # Reddish pink
#     "#66a61e",  # Olive green
#     "#e6ab02",  # Mustard yellow
#     "#a6761d",  # Brown
#     "#666666",  # Dark gray
# ]

# plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)


def plot_transient(
    ax: plt.Axes,
    data_dict: dict,
    cases=[0],
    signal_name: str = "tran_left_critical_current",
    **kwargs,
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        signal = data[signal_name]
        ax.plot(time, signal, **kwargs)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel(f"{signal_name}")
    # ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    return ax


def plot_transient_fill(
    ax: plt.Axes,
    data_dict: dict,
    cases=[0],
    s1: str = "tran_left_critical_current",
    s2: str = "tran_left_branch_current",
    **kwargs,
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        signal1 = data[s1]
        signal2 = data[s2]
        ax.fill_between(
            time,
            signal2,
            signal1,
            color=CMAP(0.5),
            alpha=0.5,
            label="Left Branch",
            **kwargs,
        )
    return ax


def plot_transient_fill_branch(
    ax: plt.Axes, data_dict: dict, cases=[0], side: Literal["left", "right"] = "left"
) -> plt.Axes:
    for i in cases:
        data_dict = data_dict[i]
        time = data_dict["time"]
        if side == "left":
            left_critical_current = data_dict["tran_left_critical_current"]
            left_branch_current = data_dict["tran_left_branch_current"]
            ax.fill_between(
                time,
                left_branch_current,
                left_critical_current,
                color=CMAP(0.5),
                alpha=0.5,
                label="Left Branch",
            )
        if side == "right":
            right_critical_current = data_dict["tran_right_critical_current"]
            right_branch_current = data_dict["tran_right_branch_current"]
            ax.fill_between(
                time,
                right_branch_current,
                right_critical_current,
                color=CMAP(0.5),
                alpha=0.5,
                label="Right Branch",
            )
    return ax


def plot_current_sweep_output(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    read_zero_voltage = data_dict["read_zero_voltage"]
    read_one_voltage = data_dict["read_one_voltage"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(
        sweep_current,
        read_zero_voltage * 1e3,
        "-o",
        label=f"{base_label} Read 0",
        **kwargs,
    )
    ax.plot(
        sweep_current,
        read_one_voltage * 1e3,
        "--o",
        label=f"{base_label} Read 1",
        **kwargs,
    )
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel(f"{sweep_param} (uA)")
    ax.legend()
    return ax


def plot_current_sweep_ber(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    ber = data_dict["bit_error_rate"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, ber, label=f"{base_label}", **kwargs)
    # ax.set_ylabel("BER")
    # ax.set_xlabel(f"{sweep_param} (uA)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_xlim(650, 850)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    return ax


def plot_current_sweep_switching(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    switching_probability = data_dict["switching_probability"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, switching_probability, label=f"{base_label}", **kwargs)
    # ax.set_ylabel("switching_probability")
    # ax.set_xlabel(f"{sweep_param} (uA)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.set_xlim(650, 850)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    return ax


def plot_current_sweep_persistent(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    persistent_current = data_dict["persistent_current"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, persistent_current, "-o", label=f"{base_label}", **kwargs)
    ax.set_ylabel("Persistent Current (uA)")
    ax.set_xlabel(f"{sweep_param} (uA)")
    return ax



def plot_case(ax, data_dict, case, signal_name="left", color=None):
    if color is None:
        if signal_name == "left":
            color = "C0"
        elif signal_name == "right":
            color = "C1"

    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_{signal_name}_critical_current",
        linestyle="--",
        color=color,
        label=f"{signal_name.capitalize()} Critical Current",
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_left_branch_current",
        color="C0",
        label="Left Branch Current",
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_right_branch_current",
        color="C1",
        label="Right Branch Current",
    )
    pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.6])


def plot_case_vout(ax, data_dict, case, signal_name, **kwargs):
    ax = plot_transient(
        ax, data_dict, cases=[case], signal_name=f"{signal_name}", **kwargs
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(50e-3))
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height / 1.6])


def create_plot(
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
            ax.set_ylim(-300, 900)
            ax.set_xlim(time_window)
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))

            ax.set_ylabel("I [µA]", labelpad=-4)
            ax.set_xlabel("Time [ns]", labelpad=-3)
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
            ax.set_xlabel("Time [ns]", labelpad=-3)
            ax.set_ylabel("V [mV]", labelpad=-3)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50e-9))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10e-9))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e9:.0f}"))

            # ax2 = ax.twinx()
            # plot_transient(ax2, data_dict, cases=[case], signal_name="tran_right_branch_current", color="C0")
    return axs


if __name__ == "__main__":

    # get raw files
    files = os.listdir("/home/omedeiro/nmem/src/nmem/analysis/read_current_sweep_sim")
    files = [f for f in files if f.endswith(".raw")]
    # Sort files by write current
    write_current_list = []
    for file in files:
        data = ltspice.Ltspice(
            f"/home/omedeiro/nmem/src/nmem/analysis/read_current_sweep_sim/{file}"
        ).parse()
        ltsp_data_dict = process_read_data(data)
        write_current = ltsp_data_dict[0]["write_current"][0]
        write_current_list.append(write_current * 1e6)

    sorted_args = np.argsort(write_current_list)
    files = [files[i] for i in sorted_args]

    data = ltspice.Ltspice(
        "/home/omedeiro/nmem/src/nmem/analysis/read_current_sweep_sim/nmem_cell_read.raw"
    ).parse()
    ltsp_data_dict = process_read_data(data)

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
        outer_nested_mosaic, figsize=(180/25.4, 180/25.4), height_ratios=[2, 0.5, 1, 1]
    )

    CASE = 16
    create_plot(axs, ltsp_data_dict, cases=[CASE])
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


    dict_list = import_directory(
        "/home/omedeiro/nmem/src/nmem/analysis/read_current_sweep_write_current2/write_current_sweep_C3"
    )
    dict_list = dict_list[::2]
    write_current_list = []
    for data_dict in dict_list:
        write_current = filter_first(data_dict["write_current"])
        write_current_list.append(write_current * 1e6)

    sorted_args = np.argsort(write_current_list)
    dict_list = [dict_list[i] for i in sorted_args]
    write_current_list = [write_current_list[i] for i in sorted_args]

    plot_read_sweep_array(
        axs["A"],
        dict_list,
        "bit_error_rate",
        "write_current",
    )
    axs["A"].set_xlim(650, 850)
    axs["A"].set_ylabel("BER")
    axs["A"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    plot_read_switch_probability_array(axs["B"], dict_list, write_current_list)
    axs["B"].set_xlim(650, 850)
    # ax.axvline(IRM, color="black", linestyle="--", linewidth=0.5)
    axs["B"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    axs["D"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)

    axs["C"].set_xlim(650, 850)
    axs["D"].set_xlim(650, 850)
    axs["C"].set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-1)
    axs["C"].set_ylabel("BER")
    axs["B"].set_ylabel("Switching Probability")
    axs["D"].set_ylabel("Switching Probability")

    # fig, ax = plt.subplots(4, 1, figsize=(6, 3))

    # plot_current_sweep_output(ax[0], data_dict)
    colors = CMAP(np.linspace(0, 1, len(data_dict)))

    for i in [0, 3, 10]:
        file = files[i]
        data = ltspice.Ltspice(
            f"/home/omedeiro/nmem/src/nmem/analysis/read_current_sweep_sim/{file}"
        ).parse()
        ltsp_data_dict = process_read_data(data)
        ltsp_write_current = ltsp_data_dict[0]["write_current"][0]
        plot_current_sweep_ber(
            axs["C"],
            ltsp_data_dict,
            color=CMAP(ltsp_write_current / 300),
            label=f"{ltsp_write_current} $\mu$A",
        )

        plot_current_sweep_switching(
            axs["D"],
            ltsp_data_dict,
            color=CMAP(ltsp_write_current / 300),
            label=f"{ltsp_write_current} $\mu$A",
        )

    axs["A"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["B"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["C"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["D"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)

    # axs["A"].legend(loc="upper left", bbox_to_anchor=(1.0, 1.05))
    axs["B"].legend(
        loc="upper right", 
        labelspacing=0.1,
        fontsize=6,
    )
    # axs["C"].legend(
    #     loc="upper right",
    # )
    axs["D"].legend(
        loc="upper right",
        labelspacing=0.1,
        fontsize=6,
    )

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.patch.set_alpha(0)


    ax_legend = fig.add_axes([0.5, 0.89, 0.1, 0.01])
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
    plt.savefig("spice_comparison.pdf", bbox_inches="tight")
    plt.show()
