import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams["figure.figsize"] = [7, 3.5]
plt.rcParams["font.size"] = 20
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


def plot_state(retrap_ratio: float, width_ratio: float, ax=None):
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

    diff = (1 + width_ratio * retrap_ratio) - (retrap_ratio + width_ratio)
    ax.fill_between(
        [-0.2, 0],
        retrap_ratio * width_ratio,
        1 + width_ratio * retrap_ratio,
        color="green",
        alpha=0.1,
    )

    ax.fill_between([0, 0.2], 1-width_ratio, retrap_ratio, color="green", alpha=0.1)

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
    plot_state(0.573, width_ratio=0.563)
    plt.show()
