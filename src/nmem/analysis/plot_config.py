import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams.update(
    {
        "figure.figsize": [3.5, 3.5],
        "font.size": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.family": "Inter",
        "lines.markersize": 2,
        "lines.linewidth": 1.2,
        "legend.fontsize": 5,
        "legend.frameon": False,
        "xtick.major.size": 1,
        "ytick.major.size": 1,
    }
)
