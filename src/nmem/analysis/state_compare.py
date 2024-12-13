import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

font_path = "/home/omedeiro/Inter-Regular.otf"

# font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"

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
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.labelpad"] = 0.5


def calculate_min_max_currents(retrap_ratio: float, width_ratio: float) -> tuple:
    imax = 1 + retrap_ratio * width_ratio
    imin = retrap_ratio + width_ratio
    return imin, imax


def plot_max_current(ax: Axes, retrap_ratio: float, width_ratio: float) -> Axes:
    imin, imax = calculate_min_max_currents(retrap_ratio, width_ratio)
    ax.hlines(imax, -0.2, 0.2, color="green", linestyle="--")
    ax.hlines(imin, -0.2, 0.2, color="green", linestyle="--")
    ax.text(-0.2, imax, "imax", ha="right", va="center", fontsize=8)
    ax.text(-0.2, imin, "imin", ha="right", va="center", fontsize=8)
    ax.fill_between([-0.2, 0.2], imin, imax, color="green", alpha=0.1)
    ax.text(0, (imax + imin) / 2, "diff", ha="center", va="center", fontsize=8)

    return ax


def plot_gap_current(ax: Axes, retrap_ratio: float, width_ratio: float) -> Axes:
    imin, imax = calculate_min_max_currents(retrap_ratio, width_ratio)
    gap = imin - 1
    ax.fill_between(
        [-0.2, 0.2],
        imin,
        1,
        color="purple",
        alpha=0.1,
    )
    ax.text(0, 1 + gap / 2, "gap", ha="center", va="center", fontsize=8)
    ax.hlines(1 + gap, -0.2, 0.2, color="purple", linestyle="--")
    return ax


def plot_branch_currents(
    ax: Axes,
    alpha: float,
    retrap_ratio: float,
    width_ratio: float,
    persistent_current: float,
    temperature: float,
):
    ax.hlines(1, 0, 1, color="red", linestyle="--")
    ax.hlines(retrap_ratio, 0, 1, color="red", linestyle=":")
    ax.hlines(width_ratio, -1, 0, color="blue", linestyle="--")
    ax.hlines(width_ratio * retrap_ratio, -1, 0, color="blue", linestyle=":")

    ax.fill_between([0, 1], retrap_ratio, 1, color="red", alpha=0.1)
    ax.fill_between(
        [-1, 0], width_ratio, width_ratio * retrap_ratio, color="blue", alpha=0.1
    )

    ax.set_xticks([-0.5, 0.5])
    ax.set_xticklabels(["Left", "Right"])
    ax.set_yticks([width_ratio * retrap_ratio, width_ratio])
    ax.set_yticklabels(["$I_{{r,H_L}}$", "$I_{{c,H_L}}$"])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.5)

    ax2 = ax.twinx()
    ax2.set_yticks([retrap_ratio, 1])
    ax2.set_ylim(0, 1.5)
    ax2.set_yticklabels(["$I_{{r,H_R}}$", "$I_{{c,H_R}}$"])

    ax.set_xlabel("hTron")
    # ax.set_ylabel("Current")
    ax.set_title(f"$r$: {retrap_ratio:.3f}, $w$: {width_ratio:.3f}")

    # diff = imax - imin

    # ax.hlines(1 - diff, 0, 1, color="green", linestyle="--")
    # ax.text(1, 1 - diff, "1-diff", ha="left", va="center", fontsize=8)
    # ax.hlines(diff, 0, 1, color="green", linestyle="--")
    # ax.text(1, diff, "diff", ha="left", va="center", fontsize=8)

    # ax.text(0, 1 + gap / 2, "gap", ha="center", va="center", fontsize=8)
    # ax.hlines(width_ratio + gap, -0.2, 0.2, color="green", linestyle="--")

    # fa = imax + diff - retrap_ratio
    # ax.hlines(fa, -0.2, 0, color="green", linestyle="--")
    # ax.text(-0.2, fa, "top-nom", ha="right", va="center", fontsize=8)

    # fb = imin + gap - diff - persistent_current
    # ax.text(-0.2, fb, "bot-nom", ha="right", va="center", fontsize=8)

    # fc = (width_ratio - persistent_current) / alpha - gap
    # ax.text(-0.2, fc, "bot-inv", ha="right", va="center", fontsize=8)

    # ax.fill_between(
    #     [-0.4, 0.4], fa, np.max([fb, fc]), color="blue", alpha=0.1, hatch="////"
    # )
    # ax.fill_between(
    #     [-0.4, 0.4],
    #     np.min([fa, np.min([fb, fc])]),
    #     np.min([np.max([fa, fc]), fb]),
    #     color="red",
    #     alpha=0.1,
    #     hatch="\\\\\\\\",
    # )

    return ax


def calculate_critical_current(T: np.ndarray, Tc: float, Ic0: float) -> np.ndarray:
    return Ic0 * (1 - (T / Tc) ** (3 / 2))


def plot_critical_current(
    ax: Axes, T: np.ndarray, Tc: float, Ic0: float, **kwargs
) -> Axes:
    Ic = calculate_critical_current(T, Tc, Ic0)
    ax.plot(T, Ic, **kwargs)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Critical Current (au)")
    return ax


def calculate_retrapping_current(
    T: np.ndarray,
    Tc: float,
    ht_coef: float,
    width: float,
    thickness: float,
    resistivity: float,
) -> np.ndarray:
    return ((ht_coef * (width**2) * Tc * thickness * resistivity) ** (1 / 2)) * (
        1 - (T / Tc)
    ) ** (1 / 2)


def calculate_retrapping_current_norm(
    T: np.ndarray, Tc: float, retrap_ratio: float
) -> np.ndarray:
    return retrap_ratio * (1 - (T / Tc)) ** (1 / 2)


def plot_retrapping_current(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    ht_coef: float,
    width: float,
    thickness: float,
    resistivity: float,
    **kwargs,
) -> Axes:
    Ic = calculate_retrapping_current(T, Tc, ht_coef, width, thickness, resistivity)
    ax.plot(T, Ic, **kwargs)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Retrapping Current (au)")
    ax.set_title("Retrapping Current vs Temperature")
    return ax


def plot_retrapping_current_norm(
    ax: Axes, T: np.ndarray, Tc: float, retrap_ratio: float, **kwargs
) -> Axes:
    Ic = calculate_retrapping_current_norm(T, Tc, retrap_ratio)
    ax.plot(T, Ic, **kwargs)
    return ax


if __name__ == "__main__":
    ALPHA = 0.563
    RETRAP = 0.573
    WIDTH = 1 / 2.13
    PERSISTENT = 0 / 860
    TEMPERATURE = 0.5
    CRITICAL_TEMP = 12.3

    retrap_list = [0.5, 0.9, RETRAP]
    width_list = [0.5, 0.9, WIDTH]
    temp = np.linspace(0, 10, 1000)
    fig, axs = plt.subplots(2, 3, figsize=(7, 4))
    for i, (retrap_ratio, width_ratio) in enumerate(zip(retrap_list, width_list)):

        plot_branch_currents(
            axs[0, i], ALPHA, retrap_ratio, width_ratio, PERSISTENT, TEMPERATURE
        )
        axs[0, i].text(0.1, 0.9, "$T=0$", transform=axs[0, i].transAxes)
        # plot_max_current(ax, retrap_ratio, width_ratio)
        # plot_gap_current(ax, retrap_ratio, width_ratio)

        plot_critical_current(
            axs[1, i], temp, CRITICAL_TEMP, 1.0, color="r", label="$I_{{c, H_R}}(T)$"
        )
        plot_retrapping_current_norm(
            axs[1, i],
            temp,
            CRITICAL_TEMP,
            retrap_ratio,
            color="r",
            ls="--",
            label="$I_{{r, H_R}}(T)$",
        )

        plot_critical_current(
            axs[1, i],
            temp,
            CRITICAL_TEMP,
            width_ratio,
            color="b",
            label="$I_{{c, H_L}}(T)$",
        )
        plot_retrapping_current_norm(
            axs[1, i],
            temp,
            CRITICAL_TEMP,
            retrap_ratio * width_ratio,
            color="b",
            ls="--",
            label="$I_{{r, H_L}}(T)$",
        )
        axs[1, i].legend()
    fig.tight_layout()
