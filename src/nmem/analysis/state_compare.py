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
