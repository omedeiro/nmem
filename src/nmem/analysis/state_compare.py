from typing import Tuple

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from nmem.measurement.cells import CELLS

# font_path = "/home/omedeiro/Inter-Regular.otf"

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 1
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.labelpad"] = 0.5


def calculate_min_max_currents(
    T: np.ndarray, Tc: float, retrap_ratio: float, width_ratio: float
) -> tuple:
    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)
    imax = ichr + irhl
    imin = ichl + irhr
    return imin, imax


def calculate_zero_temp_critical_current(Tsub: float, Tc: float, Ic: float) -> float:
    Ic0 = Ic / (1 - (Tsub / Tc) ** 3) ** (2.1)
    return Ic0


def calculate_critical_current(T: np.ndarray, Tc: float, Ic0: float) -> np.ndarray:
    return Ic0 * (1 - (T / Tc) ** (3 / 2))


def calculate_retrapping_current(
    T: np.ndarray, Tc: float, retrap_ratio: float
) -> np.ndarray:
    Ir = retrap_ratio * (1 - (T / Tc)) ** (1 / 2)
    return Ir


def calculate_branch_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ichl: np.ndarray = calculate_critical_current(T, Tc, width_ratio)
    irhl: np.ndarray = calculate_retrapping_current(T, Tc, retrap_ratio * width_ratio)
    ichr: np.ndarray = calculate_critical_current(T, Tc, 1)
    irhr: np.ndarray = calculate_retrapping_current(T, Tc, retrap_ratio)

    i0 = ichr[0] + irhl[0]

    ichl = ichl / i0
    irhl = irhl / i0
    ichr = ichr / i0
    irhr = irhr / i0
    return ichl, irhl, ichr, irhr


def calculate_state_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
) -> tuple:
    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)
    imin, imax = calculate_min_max_currents(T, Tc, retrap_ratio, width_ratio)
    fa = imax
    fb = imin - persistent_current
    fc = (ichl - persistent_current) / alpha
    fB = fb + persistent_current
    return fa, fb, fc, fB


def plot_nominal_region(
    ax: Axes, T: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, **kwargs
) -> Axes:
    ax.fill_between(
        T,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.1,
        hatch="////",
        **kwargs,
    )
    return ax


def plot_inverting_region(
    ax: Axes, T: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, **kwargs
) -> Axes:
    ax.fill_between(
        T,
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.1,
        hatch="\\\\\\\\",
        **kwargs,
    )
    return ax


def plot_state_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
    **kwargs,
):
    i0, i1, i2, _ = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )
    ax.plot(T, i0, label="$I_{{0}}$", **kwargs)
    ax.plot(T, i1, label="$I_{{1}}$", **kwargs)
    ax.plot(T, i2, label="$I_{{0,inv}}$", **kwargs)
    return ax


def plot_branch_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
) -> Axes:
    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)

    ax.plot(T, ichl, label="$I_{c, H_L}$", color="b", linestyle="-")
    ax.plot(T, irhl, label="$I_{r, H_L}$", color="b", linestyle="--")
    ax.plot(T, ichr, label="$I_{c, H_R}$", color="r", linestyle="-")
    ax.plot(T, irhr, label="$I_{r, H_R}$", color="r", linestyle="--")

    return ax


# def get_read_channel_temperature(
#     cell_dict: dict,
# ) -> dict:
#     temp_dict = {}
#     for cell in cell_dict.keys():
#         xint = cell_dict[cell].get("x_intercept")
#         x = cell_dict[cell].get("enable_read_current") * 1e6
#         temp = calculate_channel_temperature(SUBSTRATE_TEMP, CRITICAL_TEMP, x, xint)
#         read_current = cell_dict[cell].get("read_current")
#         max_critical_current = cell_dict[cell].get("max_critical_current")
#         read_current_norm = read_current / max_critical_current
#         temp_dict[cell] = {
#             "temp": temp,
#             "read_current": read_current,
#             "read_current_norm": read_current_norm,
#             "max_critical_current": max_critical_current,
#         }
#     return temp_dict


if __name__ == "__main__":
    ALPHA = 0.563
    RETRAP = 0.573
    WIDTH = 1 / 2.13
    CRITICAL_TEMP = 12.3
    SUBSTRATE_TEMP = 1.3
    IMAX = CELLS["C3"]["max_critical_current"] * 1e6
    IREAD = CELLS["C3"]["read_current"] * 1e6
    IWRITE = CELLS["C3"]["write_current"] * 1e6
    PERSISTENT = IWRITE

    temp = np.linspace(0, CRITICAL_TEMP, 1000)
    colors1 = np.flipud(plt.cm.Greens(np.linspace(0, 1, 4)))
    colors2 = np.flipud(plt.cm.Blues(np.linspace(0, 1, 4)))
    colors3 = np.flipud(plt.cm.Reds(np.linspace(0, 1, 4)))

    fig, ax = plt.subplots()
    i0, i1, i2, i3 = calculate_state_currents(
        temp, CRITICAL_TEMP, RETRAP, WIDTH, ALPHA, 0 / IMAX
    )
    ax.plot(temp, i0, label="$I_{{0}}$", color=colors1[0, :], ls="-")
    ax.plot(temp, i1, label="$I_{{1}}$", color=colors2[0, :], ls="--")
    ax.plot(temp, i2, label="$I_{{0,inv}}$", color=colors3[0, :], ls=":")

    i0, i1, i2, i3 = calculate_state_currents(
        np.array([0, 6.4]), CRITICAL_TEMP, RETRAP, WIDTH, ALPHA, PERSISTENT / IMAX
    )
    ax.plot([6.4, 6.4], [i0[1], i1[1]], marker="o", color=colors2[0, :], ls="-")
    ax.plot([6.4, 6.4], [i1[1], i2[1]], marker="o", color=colors3[0, :], ls="-")

    i0, i1, i2, i3 = calculate_state_currents(
        temp, CRITICAL_TEMP, RETRAP, WIDTH, ALPHA, PERSISTENT / IMAX
    )
    lower_bound = np.maximum(i1, i2)
    upper_bound = i0
    plot_nominal_region(ax, temp, lower_bound, upper_bound)

    i0, i1, i2, i3 = calculate_state_currents(
        temp, CRITICAL_TEMP, RETRAP, WIDTH, ALPHA, PERSISTENT / IMAX
    )
    lower_bound = np.minimum(i0, np.minimum(i1, i2))
    upper_bound = np.minimum(i1, np.maximum(i0, i2))
    plot_inverting_region(ax, temp, lower_bound, upper_bound)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Current (au)")
    ax.set_xlim([0, CRITICAL_TEMP])
    ax.set_ylim([0, 1])
