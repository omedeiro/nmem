import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes


def enable_current_relation(data, enable_current):
    # Import the data from the .mat file
    data = sio.loadmat(data)

    print(data["W1R0"])



# def plot_array(
#     data_dict: dict,
#     c_name: str,
#     x_name: str = "x",
#     y_name: str = "y",
#     ax: Axes = None,
#     cmap=None,
#     norm=True,
# ):
#     if not ax:
#         fig, ax = plt.subplots()

#     x_length = data_dict[x_name].shape[1]
#     y_length = data_dict[y_name].shape[1]

#     x = data_dict["x"][0][:, 0] * 1e6
#     y = data_dict["y"][0][:, 0] * 1e6

#     c = data_dict[c_name][0].flatten()

#     ctotal = c.reshape((len(y), len(x)), order="F")

#     dx = x[1] - x[0]
#     xstart = x[0]
#     xstop = x[-1]
#     dy = y[1] - y[0]
#     ystart = y[0]
#     ystop = y[-1]

#     if not cmap:
#         cmap = plt.get_cmap("RdBu", 100).reversed()

#     plt.imshow(
#         ctotal,
#         extent=[
#             (-0.5 * dx + xstart),
#             (0.5 * dx + xstop),
#             (-0.5 * dy + ystart),
#             (0.5 * dy + ystop),
#         ],
#         aspect="auto",
#         origin="lower",
#         cmap=cmap,
#     )

#     # print(f'{dx}, {xstart}, {xstop}, {dy}, {ystart}, {ystop}')
#     plt.xticks(np.linspace(xstart, xstop, len(x)), rotation=45)
#     plt.yticks(np.linspace(ystart, ystop, len(y)))
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     xv, yv = np.meshgrid(x, y, indexing="ij")

#     if norm:
#         plt.clim([0, 1])
#         cbar = plt.colorbar(ticks=np.linspace(0, 1, 11))
#     else:
#         plt.clim([0, c.max()])
#         cbar = plt.colorbar()
#     cbar.ax.set_ylabel(c_name, rotation=270)
#     plt.contour(xv, yv, np.reshape(ctotal, (len(y), len(x))).T, [0.05, 0.1])
#     measurement_name = data_dict["measurement_name"].flatten()
#     sample_name = data_dict["sample_name"].flatten()
#     time_str = data_dict["time_str"]

#     plt.suptitle(f"{sample_name[0]} -- {measurement_name[0]} \n {time_str}")

#     return ax

if __name__ == "__main__":
    data = "data\SPG806_20240801_measure_enable_response_D6_A4_2024-08-01 14-32-23.mat"
    enable_current = 0.0
    enable_current_relation(data, enable_current)
