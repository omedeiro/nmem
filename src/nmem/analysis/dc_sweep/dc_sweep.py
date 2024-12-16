from typing import Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from nmem.analysis.analysis import import_directory

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


def plot_iv_curve(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    time = data_dict.get("trace")[0, :]
    voltage = data_dict.get("trace")[1, :]

    M = int(np.round(len(voltage), -2))
    currentQuart = np.linspace(0, data_dict["vpp"] / 2 / 10e3, M // 4)
    current = np.concatenate(
        [-currentQuart, np.flip(-currentQuart), currentQuart, np.flip(currentQuart)]
    )

    if len(voltage) > M:
        voltage = voltage[:M]
        time = time[:M]
    else:
        voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
        time = np.concatenate([time, np.zeros(M - len(time))])

    ax.plot(voltage, current.flatten() * 1e6, **kwargs)
    return ax


def plot_iv(ax: Axes, data_list: list, save: bool = False) -> Axes:
    colors = plt.cm.coolwarm(np.linspace(0, 1, int(len(data_list) / 2) + 1))
    colors = np.flipud(colors)
    for i, data in enumerate(data_list):
        heater_current = np.abs(data["heater_current"].flatten()[0] * 1e6)
        ax = plot_iv_curve(
            ax, data, color=colors[i], zorder=-i, label=f"{heater_current:.0f} µA"
        )
        if i == 10:
            break
    ax.set_ylim([-500, 500])
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [µA]", labelpad=-1)
    ax.tick_params(direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.legend(frameon=False, handlelength=0.5, labelspacing=0.1)

    if save:
        plt.savefig("iv_curve.pdf", bbox_inches="tight")

    return ax


def get_critical_currents(data_list: list) -> Tuple[list, list]:
    critical_currents = []
    critical_currents_std = []
    for data in data_list:
        time = data.get("trace")[0, :]
        voltage = data.get("trace")[1, :]

        M = int(np.round(len(voltage), -2))
        if len(voltage) > M:
            voltage = voltage[:M]
            time = time[:M]
        else:
            voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
            time = np.concatenate([time, np.zeros(M - len(time))])

        current_time_trend = (
            data["vpp"]
            / 2
            / 10e3
            * (data["time_trend"][1, :])
            / (1 / (data["freq"] * 4))
            * 1e6
        )

        avg_critical_current = np.mean(current_time_trend)
        std_critical_current = np.std(current_time_trend)
        critical_currents.append(avg_critical_current)
        critical_currents_std.append(std_critical_current)

    return critical_currents, critical_currents_std


def plot_critical_currents(ax: Axes = None, data_list: list = None) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents(data_list)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(data_list)))
    heater_currents = [data["heater_current"].flatten() * 1e6 for data in data_list]
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


def plot_critical_currents_abs(ax: Axes, data_list: list, save: bool = False) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents(data_list)

    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(data_list)))
    heater_currents = np.array(
        [data["heater_current"].flatten() * 1e6 for data in data_list]
    ).flatten()
    positive_critical_currents = np.where(
        heater_currents > 0, critical_currents, np.nan
    )
    negative_critical_currents = np.where(
        heater_currents < 0, critical_currents, np.nan
    )
    ax.plot(
        np.abs(heater_currents),
        positive_critical_currents,
        "o--",
        color=cmap[0, :],
        label="$+I_{{h}}$",
        linewidth=0.5,
        markersize=0.5,
        markerfacecolor=cmap[0, :],
    )
    ax.plot(
        np.abs(heater_currents),
        negative_critical_currents,
        "o--",
        color=cmap[-5, :],
        label="$-I_{{h}}$",
        linewidth=0.5,
        markersize=0.5,
        markerfacecolor=cmap[-5, :],
    )
    ax.fill_between(
        np.abs(heater_currents),
        (positive_critical_currents + critical_currents_std),
        (positive_critical_currents - critical_currents_std),
        color=cmap[0, :],
        alpha=0.3,
        edgecolor="none",
    )
    ax.fill_between(
        np.abs(heater_currents),
        (negative_critical_currents + critical_currents_std),
        (negative_critical_currents - critical_currents_std),
        color=cmap[-5, :],
        alpha=0.3,
        edgecolor="none",
    )
    ax.tick_params(direction="in", top=True, right=True, bottom=True, left=True)
    ax.set_xlabel("$|I_{{h}}|$[$\mu$A]")

    ax.set_ylabel("$I_{{C}}$ [$\mu$A]", labelpad=-1)
    ax.set_ylim([0, 400])
    ax.set_xlim([0, 500])
    ax.legend(frameon=False)

    if save:
        plt.savefig("critical_currents_abs.pdf", bbox_inches="tight")

    return ax


def plot_critical_currents_inset(ax: Axes, data_list: list, save: bool = False) -> Axes:
    ax = plot_iv(ax, data_list)
    fig = plt.gcf()
    ax_inset = fig.add_axes([0.62, 0.25, 0.3125, 0.25])
    ax_inset = plot_critical_currents_abs(ax_inset, data_list)
    ax_inset.xaxis.tick_top()
    ax_inset.tick_params(direction="in", top=True, right=True, bottom=True, left=True)

    ax_inset.xaxis.set_label_position("top")
    ax_inset.xaxis.set_major_locator(MultipleLocator(0.1))

    if save:
        plt.savefig("critical_currents_inset.pdf", bbox_inches="tight")

    return ax


def plot_combined_figure(ax: Axes, data_list: list, save: bool = False) -> Axes:
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 0].axis("off")
    ax[1, 1] = plot_iv(ax[1, 1], data_list)
    ax[1, 2] = plot_critical_currents_abs(ax[1, 2], data_list)
    plt.subplots_adjust(wspace=0.3)
    fig.patch.set_visible(False)

    if save:
        plt.savefig("iv_curve_combined.pdf", bbox_inches="tight")

    return ax


if __name__ == "__main__":
    data_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\dc_sweep"
    )

    fig, ax = plt.subplots()
    plot_critical_currents_abs(ax, data_list)

    fig, ax = plt.subplots()
    plot_iv(ax, data_list)

    fig, ax = plt.subplots(2, 3, figsize=(7, 3.5), height_ratios=[1, 0.7])
    plot_combined_figure(ax, data_list)
