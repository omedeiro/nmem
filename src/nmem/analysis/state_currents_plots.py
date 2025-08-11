import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import Literal
from nmem.analysis.calculated_plots import (
    plot_calculated_filled_region,
)
from nmem.analysis.constants import (
    ALPHA,
    CRITICAL_TEMP,
    RETRAP,
    WIDTH,
)
from nmem.analysis.currents import (
    calculate_branch_currents,
    calculate_state_currents,
    get_state_currents_measured,
)
from nmem.analysis.utils import (
    filter_nan,
)
from nmem.measurement.functions import (
    calculate_power,
)
from nmem.analysis.styles import CMAP


def get_step_parameter(data_dict: dict) -> str:
    keys = [
        "write_current",
        "read_current",
        "enable_write_current",
        "enable_read_current",
    ]
    for key in keys:
        data = data_dict[key]
        if data[0] != data[1]:
            return key
    return None


def plot_branch_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    critical_current_zero: float,
) -> Axes:
    ichl, irhl, ichr, irhr = calculate_branch_currents(
        T, Tc, retrap_ratio, width_ratio, critical_current_zero
    )

    ax.plot(T, ichl, label="$I_{c, H_L}(T)$", color="b", linestyle="-")
    ax.plot(T, irhl, label="$I_{r, H_L}(T)$", color="b", linestyle="--")
    ax.plot(T, ichr, label="$I_{c, H_R}(T)$", color="r", linestyle="-")
    ax.plot(T, irhr, label="$I_{r, H_R}(T)$", color="r", linestyle="--")

    ax.plot(T, ichr + irhl, label="$I_{0}(T)$", color="g", linestyle="-")
    return ax


def plot_calculated_state_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
    critical_current_zero: float,
    **kwargs,
):
    i0, i1, i2, i3 = calculate_state_currents(
        T,
        Tc,
        retrap_ratio,
        width_ratio,
        alpha,
        persistent_current,
        critical_current_zero,
    )
    ax.plot(T, i0, label="$I_{{0}}(T)$", **kwargs)
    ax.plot(T, i1, label="$I_{{1}}(T)$", **kwargs)
    ax.plot(T, i2, label="$I_{{0,inv}}(T)$", **kwargs)
    ax.plot(T, i3, label="$I_{{1,inv}}(T)$", **kwargs)
    return ax


def plot_state_currents_measured_nominal(
    ax: Axes, nominal_read_temperature_list: list, nominal_state_currents_list: list
) -> Axes:
    for t, temp in enumerate(nominal_read_temperature_list):
        ax.plot(
            [temp, temp],
            nominal_state_currents_list[t],
            "o",
            linestyle="-",
            color="blue",
        )
    return ax


def plot_state_currents_measured_inverting(
    ax: Axes, inverting_read_temperature_list: list, inverting_state_currents_list: list
) -> Axes:
    for t, temp in enumerate(inverting_read_temperature_list):
        ax.plot(
            [temp, temp],
            inverting_state_currents_list[t],
            "o",
            linestyle="-",
            color="red",
        )
    return ax


def plot_state_currents_measured(ax: Axes, data_dict: dict, current_sweep: str) -> Axes:
    temp, state_currents = get_state_currents_measured(data_dict, current_sweep)

    if state_currents[0] is not np.nan:
        ax.plot(
            [temp, temp],
            state_currents[0:2],
            "o",
            linestyle="-",
            color="blue",
            label="_state0",
        )
    if state_currents[2] is not np.nan:
        ax.plot(
            [temp, temp],
            state_currents[2:4],
            "o",
            linestyle="-",
            color="red",
            label="_state1",
        )

    return ax


def plot_state_current_fit(ax, x_list, y_list, x_list_full, model, colors):
    for i in range(4):
        ax.plot(x_list[i], y_list[i], "-o", color=colors[i], label=f"State {i}")
        ax.plot(x_list_full, model[i], "--", color=colors[i])
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [$\\mu$A]")
    ax.grid()
    ax.set_ybound(lower=0)
    ax.set_xbound(lower=0)


def plot_measured_state_current_list(ax: Axes, dict_list: list[dict]) -> Axes:
    sweep_length = len(dict_list)
    for j in range(0, sweep_length):
        plot_state_currents_measured(ax, dict_list[j], "enable_read_current")

    return ax


def plot_persistent_current(
    data_dict, persistent_current=75, critical_current_zero=1250
):
    """
    Plots calculated and measured persistent current curves for a given data_dict.
    Returns (fig, ax).
    """
    power = calculate_power(data_dict)
    fig, ax = plt.subplots()
    temperatures = np.linspace(0, CRITICAL_TEMP, 100)
    plot_calculated_state_currents(
        ax,
        temperatures,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        persistent_current,
        critical_current_zero,
    )
    plot_calculated_filled_region(
        ax,
        temperatures,
        data_dict,
        persistent_current,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        critical_current_zero,
    )
    return fig, ax


def plot_simulation_results(axs, ltsp_data_dict, case=16):
    """Plot simulation results for a given case on provided axes."""

    create_plot(axs, ltsp_data_dict, cases=[case])
    handles, labels = axs["T0"].get_legend_handles_labels()
    selected_labels = [
        "Left Branch Current",
        "Right Branch Current",
        "Left Critical Current",
        "Right Critical Current",
    ]
    selected_labels2 = [
        "$i_{\\mathrm{H_L}}$",
        "$i_{\\mathrm{H_R}}$",
        "$I_{\\mathrm{c,H_L}}$",
        "$I_{\\mathrm{c,H_R}}$",
    ]
    selected_handles = [handles[labels.index(lbl)] for lbl in selected_labels]
    return selected_handles, selected_labels2


def plot_measured_state_currents(ax, mat_files, colors):
    """
    Plots measured state currents from a list of .mat files on the given axis.
    """
    for data_dict in mat_files:
        temp = data_dict["measured_temperature"].flatten()
        state_currents = data_dict["measured_state_currents"]
        for i in range(4):
            x, y = filter_nan(temp, state_currents[:, i])
            ax.plot(x, y, "-o", color=colors[i], label=f"State {i}")
    return ax


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
        linewidth=1.5,
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name="tran_left_branch_current",
        color="C0",
        label="Left Branch Current",
        linewidth=1.5,
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name="tran_right_branch_current",
        color="C1",
        label="Right Branch Current",
        linewidth=1.5,
    )
    pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 1.6])


def plot_case_vout(ax, data_dict, case, signal_name, **kwargs):
    ax = plot_transient(
        ax, data_dict, cases=[case], signal_name=f"{signal_name}", **kwargs
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(50e-3))
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height / 1.6])


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
                linewidth=1.5,
            )
            ax.plot(
                data_dict[case]["time"],
                -1 * data_dict[case]["tran_right_critical_current"],
                color="C1",
                linestyle="--",
                linewidth=1.5,
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
            plot_case_vout(
                ax, data_dict, case, "tran_output_voltage", color="k", linewidth=1.5
            )
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
