import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.labelpad"] = 0.5


def plot_state(
    alpha,
    retrap_ratio: float,
    width_ratio: float,
    persistent_current: float,
    temperature: float,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    x = np.linspace(0, 1, 100)
    ax.hlines(1, 0, 1, color="red", linestyle="--")
    ax.hlines(retrap_ratio, 0, 1, color="red", linestyle=":")
    ax.hlines(width_ratio, -1, 0, color="blue", linestyle="--")
    ax.hlines(width_ratio * retrap_ratio, -1, 0, color="blue", linestyle=":")

    ax.fill_between([0, 1], retrap_ratio, 1, color="red", alpha=0.1)
    ax.fill_between(
        [-1, 0], width_ratio, width_ratio * retrap_ratio, color="blue", alpha=0.1
    )

    imax = 1 + retrap_ratio * width_ratio
    imin = retrap_ratio + width_ratio
    ax.hlines(imax, -0.2, 0.2, color="green", linestyle="--")
    ax.hlines(imin, -0.2, 0.2, color="green", linestyle="--")
    ax.text(0.2, imax, "imax", ha="left", va="center", fontsize=8)
    ax.text(0.2, imin, "imin", ha="left", va="center", fontsize=8)
    ax.fill_between([-0.2, 0.2], imin, imax, color="green", alpha=0.1)
    ax.text(0, (imax + imin) / 2, "diff", ha="center", va="center", fontsize=8)

    diff = imax - imin

    ax.hlines(1 - diff, 0, 1, color="green", linestyle="--")
    ax.text(1, 1 - diff, "1-diff", ha="left", va="center", fontsize=8)
    ax.hlines(diff, 0, 1, color="green", linestyle="--")
    ax.text(1, diff, "diff", ha="left", va="center", fontsize=8)

    gap = imin - 1
    ax.fill_between(
        [-0.2, 0.2],
        imin,
        1,
        color="purple",
        alpha=0.1,
    )
    ax.text(0, 1 + gap / 2, "gap", ha="center", va="center", fontsize=8)
    ax.hlines(width_ratio + gap, -0.2, 0.2, color="green", linestyle="--")

    # ax.fill_between([0, 0.2], 1-width_ratio, retrap_ratio, color="green", alpha=0.1)
    fa = imax + diff - retrap_ratio
    ax.hlines(fa, -0.2, 0, color="green", linestyle="--")
    ax.text(-0.2, fa, "top-nom", ha="right", va="center", fontsize=8)

    # fb = (
    #     width_ratio
    #     + retrap_ratio
    #     + (gap-diff) # Yb = Q-q = gap-diff
    #     - persistent_current
    # )
    # fb = (
    #     width_ratio
    #     + retrap_ratio
    #     + retrap_ratio
    #     - (1 - width_ratio)
    #     - 1
    #     - width_ratio * retrap_ratio
    #     + retrap_ratio
    #     + width_ratio
    #     - persistent_current
    # )
    fb = imin + gap - diff - persistent_current
    # ax.hlines(fb, 0, 0.2, color="blue", linestyle="--")
    ax.text(-0.2, fb, "bot-nom", ha="right", va="center", fontsize=8)

    fc = (width_ratio - persistent_current) / alpha - gap
    # ax.hlines(fc, -0.2, 0, color="blue", linestyle="--")
    ax.text(-0.2, fc, "bot-inv", ha="right", va="center", fontsize=8)

    ax.fill_between(
        [-0.4, 0.4], fa, np.max([fb, fc]), color="blue", alpha=0.1, hatch="////"
    )
    ax.fill_between(
        [-0.4, 0.4],
        np.min([fa, np.min([fb, fc])]),
        np.min([np.max([fa, fc]), fb]),
        color="red",
        alpha=0.1,
        hatch="\\\\\\\\",
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
    ax.set_ylabel("Current")
    ax.set_title(f"$r$: {retrap_ratio:.3f}, $w$: {width_ratio:.3f}")
    return ax


if __name__ == "__main__":
    ALPHA = 0.563
    RETRAP = 0.573
    WIDTH = 1 / 2.13
    PERSISTENT = 0 / 860
    TEMPERATURE = 0.5
    plot_state(ALPHA, RETRAP, WIDTH, PERSISTENT, TEMPERATURE)
    plt.show()
