import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb
from matplotlib import colors as mcolors

RBCOLORS = {0: "blue", 1: "blue", 2: "red", 3: "red"}
C0 = "#1b9e77"
C1 = "#d95f02"
CMAP = plt.get_cmap("coolwarm")
CMAP2 = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [C0, C1])
CMAP3 = plt.get_cmap("plasma").reversed()

def set_pres_style(dpi=600, font_size=14, grid_alpha=0.4):
    """
    Apply a presentation-optimized Matplotlib style.

    Parameters:
        dpi (int): Figure DPI (for saved files).
        font_size (int): Base font size for axes and labels.
        grid_alpha (float): Grid line transparency.
    """
    set_inter_font()
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "font.family": "Inter",
            "figure.figsize": (6, 4),
            "axes.titlesize": font_size + 4,
            "axes.labelsize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "font.size": font_size,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": grid_alpha,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )

def darken(color, factor=0.6):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))


def lighten(color, factor=1.1):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))



def set_inter_font():
    if os.name == "nt":  # Windows
        font_path = r"C:\Users\ICE\AppData\Local\Microsoft\Windows\Fonts\Inter-VariableFont_opsz,wght.ttf"
    elif os.name == "posix":
        font_path = "/home/omedeiro/Inter-VariableFont_opsz,wght.ttf"
    else:
        font_path = None

    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "Inter"


def set_plot_style() -> None:
    set_inter_font()
    golden_ratio = (1 + 5**0.5) / 2  # ≈1.618
    width = 3.5  # Example width in inches (single-column for Nature)
    height = width / golden_ratio
    plt.rcParams.update(
        {
            "figure.figsize": [width, height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.markersize": 3,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )

