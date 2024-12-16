import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import Tuple

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


def calculate_min_max_currents(
    T: np.ndarray, Tc: float, retrap_ratio: float, width_ratio: float
) -> tuple:
    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)
    imax = ichr + irhl
    imin = ichl + irhr
    return imin, imax


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
    return ichl, irhl, ichr, irhr


def calculate_offsets(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)
    imin, imax = calculate_min_max_currents(T, Tc, retrap_ratio, width_ratio)
    gap = imin - ichr
    diff = imax - ichl
    return gap, diff


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
    gap, diff = calculate_offsets(T, Tc, retrap_ratio, width_ratio)
    if isinstance(gap, np.ndarray):
        gap = gap[0]
        diff = diff[0]
        irhr = irhr[0]
    fa = imax
    fb = imin - persistent_current
    fc = (ichl - persistent_current) / alpha
    fB = fb + persistent_current
    return fa, fb, fc, fB


def plot_critical_current(
    ax: Axes, T: np.ndarray, Ic0: float, data_dict: dict, **kwargs
) -> Axes:
    Tc = data_dict.get("critical_temp")
    Ic = calculate_critical_current(T, Tc, Ic0)
    ax.plot(T, Ic, **kwargs)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Critical Current (au)")
    return ax


def plot_retrapping_current(
    ax: Axes,
    T: np.ndarray,
    retrap_ratio: float,
    data_dict: dict,
    **kwargs,
) -> Axes:
    Tc = data_dict.get("critical_temp")
    Ir = calculate_retrapping_current(T, Tc, retrap_ratio)
    ax.plot(T, Ir, **kwargs)
    return ax


def plot_max_current(ax: Axes, T: np.ndarray, data_dict: dict) -> Axes:
    Tc = data_dict.get("critical_temp")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    imin, imax = calculate_min_max_currents(T, Tc, retrap_ratio, width_ratio)
    ax.hlines(imax, *ax.get_xlim(), color="green", linestyle="--")
    ax.hlines(imin, *ax.get_xlim(), color="green", linestyle="--")
    ax.text(-0.2, imax, "imax", ha="right", va="center", fontsize=8)
    ax.text(-0.2, imin, "imin", ha="right", va="center", fontsize=8)
    ax.fill_between(ax.get_xlim(), imin, imax, color="green", alpha=0.1)
    ax.text(0, (imax + imin) / 2, "diff", ha="center", va="center", fontsize=8)

    return ax


def plot_min_max_currents(ax: Axes, T: np.ndarray, data_dict: dict) -> Axes:
    Tc = data_dict.get("critical_temp")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    imin, imax = calculate_min_max_currents(T, Tc, retrap_ratio, width_ratio)
    ax.plot(T, imin, label="imin")
    ax.plot(T, imax, label="imax")
    return ax


def plot_nominal_region(ax: Axes, T: np.ndarray, data_dict: dict) -> Axes:
    Tc = data_dict.get("critical_temp")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    persistent_current = data_dict.get("persistent_current")
    alpha = data_dict.get("alpha")
    fa, fb, fc, fB = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )
    lower_bound = np.maximum(fb, fc)
    upper_bound = fa
    ax.fill_between(
        T,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.1,
        hatch="////",
    )
    return ax


def plot_inverting_region(ax: Axes, T: np.ndarray, data_dict: dict) -> Axes:
    Tc = data_dict.get("critical_temp")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    persistent_current = data_dict.get("persistent_current")
    alpha = data_dict.get("alpha")
    fa, fb, fc, fB = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )
    lower_bound = np.minimum(fa, np.minimum(fB, fc))  # big Z
    upper_bound = np.minimum(np.maximum(fa, fc), fb)
    ax.fill_between(
        T,
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.1,
        hatch="\\\\\\\\",
    )
    return ax


def plot_state_currents_line(
    ax: Axes,
    T: np.ndarray,
    data_dict: dict,
):
    alpha = data_dict.get("alpha")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    persistent_current = data_dict.get("persistent_current")
    Tc = data_dict.get("critical_temp")

    i0, i1, i2, i3 = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )

    ax.plot(T, i0, label="State 0", color="k", ls="-")
    ax.plot(T, i1, label="State 1", color="k", ls="--")
    ax.plot(T, i2, label="State 2", color="k", ls=":")
    return ax


def plot_state_currents(
    ax: Axes,
    T: float,
    data_dict: dict,
) -> Axes:
    alpha = data_dict.get("alpha")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    persistent_current = data_dict.get("persistent_current")
    Tc = data_dict.get("critical_temp")

    fa, fb, fc, fB = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )
    ax.hlines(fa, ax.get_xlim()[0], ax.get_xlim()[1], color="black", linestyle="-")
    ax.hlines(fb, ax.get_xlim()[0], ax.get_xlim()[1], color="black", linestyle="--")
    ax.hlines(fc, ax.get_xlim()[0], ax.get_xlim()[1], color="black", linestyle=":")
    return ax


def plot_branch_currents(
    ax: Axes,
    T: float,
    data_dict: dict,
):
    alpha = data_dict.get("alpha")
    retrap_ratio = data_dict.get("retrap_ratio")
    width_ratio = data_dict.get("width_ratio")
    persistent_current = data_dict.get("persistent_current")
    Tc = data_dict.get("critical_temp")

    ichl, irhl, ichr, irhr = calculate_branch_currents(T, Tc, retrap_ratio, width_ratio)
    ax.hlines(ichr, 0, 1, color="red", linestyle="--")
    ax.hlines(irhr, 0, 1, color="red", linestyle=":")
    ax.hlines(ichl, -1, 0, color="blue", linestyle="--")
    ax.hlines(irhl, -1, 0, color="blue", linestyle=":")

    # ax.fill_between([0, 1], retrap_ratio, 1, color="red", alpha=0.1)
    # ax.fill_between(
    #     [-1, 0], width_ratio, width_ratio * retrap_ratio, color="blue", alpha=0.1
    # )

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

    fa, fb, fc, fB = calculate_state_currents(
        T, Tc, retrap_ratio, width_ratio, alpha, persistent_current
    )

    ax.fill_between(
        ax.get_xlim(), fa, np.max([fb, fc]), color="blue", alpha=0.1, hatch="////"
    )
    ax.fill_between(
        ax.get_xlim(),
        np.min([fa, np.min([fb, fc])]),
        np.min([np.max([fa, fc]), fb]),
        color="red",
        alpha=0.1,
        hatch="\\\\\\\\",
    )

    return ax


# def calculate_retrapping_current(
#     T: np.ndarray,
#     Tc: float,
#     ht_coef: float,
#     width: float,
#     thickness: float,
#     resistivity: float,
# ) -> np.ndarray:
#     return ((ht_coef * (width**2) * Tc * thickness * resistivity) ** (1 / 2)) * (
#         1 - (T / Tc)
#     ) ** (1 / 2)


def create_data_dict(
    alpha: float,
    retrap_ratio: float,
    width_ratio: float,
    persistent_current: float,
    Tc: float,
):
    return {
        "alpha": alpha,
        "retrap_ratio": retrap_ratio,
        "width_ratio": width_ratio,
        "persistent_current": persistent_current,
        "critical_temp": Tc,
    }


if __name__ == "__main__":
    ALPHA = 0.563
    RETRAP = 0.573
    WIDTH = 1 / 2.13
    PERSISTENT = 30 / 860
    CRITICAL_TEMP = 12.3

    data_dict = create_data_dict(ALPHA, RETRAP, WIDTH, PERSISTENT, CRITICAL_TEMP)

    retrap_list = [0.5, 0.5, RETRAP]
    width_list = [0.5, 0.33, WIDTH]
    persistent_list = [0.1, 0.1, 0.1]
    temp = np.linspace(0, 10, 1000)
    fig, axs = plt.subplots(2, 3, figsize=(7, 4))
    for i, (retrap_ratio, width_ratio) in enumerate(zip(retrap_list, width_list)):
        data_dict["retrap_ratio"] = retrap_ratio
        data_dict["width_ratio"] = width_ratio
        data_dict["persistent_current"] = persistent_list[i]
        plot_branch_currents(
            axs[0, i],
            temp[0],
            data_dict,
        )
        plot_state_currents(axs[0, i], temp[0], data_dict)

        # plot_max_current(axs[0, i], temp[0], data_dict)
        axs[0, i].text(0.1, 0.9, "$T=0$", transform=axs[0, i].transAxes)

        label_list = [
            "$I_{{c, H_R}}(T)$",
            "$I_{{r, H_R}}(T)$",
            "$I_{{c, H_L}}(T)$",
            "$I_{{r, H_L}}(T)$",
        ]
        color_list = ["r", "r", "b", "b"]
        linestyle_list = ["-", "--", "-", "--"]
        Ic0_list = [1, retrap_ratio, width_ratio, width_ratio * retrap_ratio]
        for j in range(4):
            if j % 2 == 0:
                plot_critical_current(
                    axs[1, i],
                    temp,
                    Ic0_list[j],
                    data_dict,
                    label=label_list[j],
                    color=color_list[j],
                    linestyle=linestyle_list[j],
                )
            else:
                plot_retrapping_current(
                    axs[1, i],
                    temp,
                    Ic0_list[j],
                    data_dict,
                    label=label_list[j],
                    color=color_list[j],
                    linestyle=linestyle_list[j],
                )

        plot_state_currents_line(axs[1, i], temp, data_dict)
        # plot_min_max_currents(axs[1, i], temp, data_dict)
        axs[1, 0].legend(loc="upper left", ncol=3, fontsize=5)
        plot_nominal_region(axs[1, i], temp, data_dict)
        plot_inverting_region(axs[1, i], temp, data_dict)
        axs[1, i].set_ylim(0, 1.5)
        axs[1, i].text(
            0.1,
            0.08,
            f"$I_{{P}}: {persistent_list[i]:.2f}I_{{c}}$",
            transform=axs[1, i].transAxes,
        )
    fig.tight_layout()
    plt.savefig("state_currents.pdf", bbox_inches="tight")
