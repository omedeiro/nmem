from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nmem.analysis.core_analysis import get_fitting_points
from nmem.analysis.currents import (
    get_channel_temperature,
    get_critical_currents_from_trace,
    get_current_cell,
    get_enable_current_sweep,
)
from nmem.analysis.plot_utils import (
    plot_fitting,
)
from nmem.analysis.styles import CMAP
from nmem.analysis.utils import (
    build_array,
    convert_cell_to_coordinates,
)


def plot_c2c3_comparison(ax, c2, c3, split_idx=10):
    """
    Plot C2 and C3 comparison on a single axis.
    """
    xfit, yfit = c2
    ax.plot(xfit, yfit, label="C2", linestyle="-")
    plot_fitting(
        ax,
        xfit[split_idx + 1 :],
        yfit[split_idx + 1 :],
        label="_C2",
        linestyle="-",
        add_text=True,
    )
    xfit, yfit = c3
    ax.plot(xfit, yfit, label="C3", linestyle="-")
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 500])
    ax.set_xlabel("Enable Current ($\mu$A)")
    ax.set_ylabel("Critical Current ($\mu$A)")
    ax.legend()
    return ax


def plot_c3_subplots(axs, c3, split_idx):
    """
    Plot C2 and C3 comparison on two subplots.
    """
    xfit, yfit = c3
    plot_fitting(
        axs[0],
        xfit[split_idx + 1 :],
        yfit[split_idx + 1 :],
        label="C3",
        linestyle="-",
        add_text=True,
    )
    axs[0].plot(xfit, yfit, label="C3", linestyle="-")
    axs[0].set_ylim([0, 1000])
    axs[0].set_xlim([0, 500])
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    axs[0].legend()
    plot_fitting(
        axs[1],
        xfit[:split_idx],
        yfit[:split_idx],
        label="C3",
        linestyle="-",
        add_text=True,
    )
    axs[1].plot(xfit, yfit, label="C3", linestyle="-")
    axs[1].set_ylim([0, 1000])
    axs[1].set_xlim([0, 500])
    axs[1].set_xlabel("Enable Current ($\mu$A)")
    axs[1].legend()
    return axs


def plot_critical_currents_from_trace(ax: Axes, dict_list: list) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents_from_trace(
        dict_list
    )
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(dict_list)))
    heater_currents = [data["heater_current"].flatten() * 1e6 for data in dict_list]
    ax.errorbar(
        heater_currents,
        critical_currents,
        yerr=critical_currents_std,
        fmt="o",
        markersize=3,
        color=cmap[0, :],
    )
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Heater Current [µA]")
    ax.set_ylabel("Critical Current [µA]")
    ax.set_ylim([0, 400])
    return ax


def plot_enable_current_relation(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        x, y, ztotal = build_array(data_dict, "total_switches_norm")
        xfit, yfit = get_fitting_points(x, y, ztotal)
        ax.plot(xfit, yfit, label=f"{cell}", color=colors[column], marker=markers[row])
    return ax


def plot_channel_temperature(
    ax: plt.Axes,
    data_dict: dict,
    channel_sweep: Literal["enable_write_current", "enable_read_current"],
    **kwargs,
) -> plt.Axes:
    if channel_sweep == "enable_write_current":
        temp = get_channel_temperature(data_dict, "write")
        current = get_enable_current_sweep(data_dict)
        print(f"temp: {temp}, current: {current}")
    else:
        temp = get_channel_temperature(data_dict, "read")
        current = get_enable_current_sweep(data_dict)
        print(f"temp: {temp}, current: {current}")
    ax.plot(current, temp, **kwargs)

    return ax
