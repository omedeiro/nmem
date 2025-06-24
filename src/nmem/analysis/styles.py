import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb

RBCOLORS = {0: "blue", 1: "blue", 2: "red", 3: "red"}
C0 = "#1b9e77"
C1 = "#d95f02"
CMAP = plt.get_cmap("coolwarm")
CMAP2 = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [C0, C1])
CMAP3 = plt.get_cmap("plasma").reversed()

# Global style configuration
STYLE_CONFIG = {
    "mode": "paper",  # Can be "presentation", "paper", or "thesis"
}


def set_style_mode(mode: str) -> None:
    """
    Set the global plotting style mode.

    Args:
        mode (str): Style mode. Options are:
            - "presentation" or "pres": For presentations and talks
            - "paper" or "publication": For academic papers and publications
            - "thesis": For thesis writing
    """
    valid_modes = {
        "presentation": "presentation",
        "pres": "presentation",
        "paper": "paper",
        "publication": "paper",
        "thesis": "thesis",
    }

    if mode.lower() not in valid_modes:
        raise ValueError(
            f"Invalid style mode '{mode}'. Valid options are: {list(valid_modes.keys())}"
        )

    STYLE_CONFIG["mode"] = valid_modes[mode.lower()]
    print(f"Global plot style set to: {STYLE_CONFIG['mode']}")


def get_style_mode() -> str:
    """Get the current global plotting style mode."""
    return STYLE_CONFIG["mode"]


def apply_global_style(**kwargs) -> None:
    """
    Apply the globally configured plotting style.

    Args:
        **kwargs: Additional style parameters to override defaults
    """
    mode = STYLE_CONFIG["mode"]

    if mode == "presentation":
        set_pres_style(**kwargs)
    elif mode == "paper":
        set_paper_style(**kwargs)
    elif mode == "thesis":
        set_thesis_style(**kwargs)
    else:
        # Default to thesis style
        set_thesis_style(**kwargs)


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


def set_paper_style() -> None:
    """Paper/publication-optimized style with Inter fonts and compact layout."""
    set_inter_font()
    golden_ratio = (1 + 5**0.5) / 2  # ≈1.618
    width = 3.5  # Example width in inches (single-column for Nature)
    height = width / golden_ratio
    plt.rcParams.update(
        {
            "figure.figsize": [width, height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Inter",
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


def set_thesis_style(dpi=300, font_size=11, grid_alpha=0.3) -> None:
    """
    Thesis-optimized style - balanced between presentation and paper styles.

    Parameters:
        dpi (int): Figure DPI (for saved files).
        font_size (int): Base font size for axes and labels.
        grid_alpha (float): Grid line transparency.
    """
    golden_ratio = (1 + 5**0.5) / 2  # ≈1.618
    width = 5.0  # Slightly larger than paper style
    height = width / golden_ratio

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "figure.figsize": [width, height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.markersize": 4,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": grid_alpha,
            "axes.edgecolor": "#333333",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )
