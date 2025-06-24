import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

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
from nmem.simulation.spice_circuits.plotting import (
    create_plot,
)


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
