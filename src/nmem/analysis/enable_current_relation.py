import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes


def find_peak(data: dict):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6

    w0r1 = 100 - data["write_0_read_1"][0].flatten()
    w1r0 = data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    mid = np.where((ztotal > 10) * (ztotal < 190), ztotal, 0)
    mid_idx = np.where(mid > 0)

    plt.scatter(x[mid_idx[1]], y[mid_idx[0]])

    # Plot a fit line to the scatter points
    z = np.polyfit(x[mid_idx[1]], y[mid_idx[0]], 1)
    p = np.poly1d(z)

    print(f"slope: {z[0]}, intercept: {z[1]}")
    plt.plot(x[mid_idx[1]], p(x[mid_idx[1]]), "r--")

    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    cbar = plt.colorbar()

    # Add fit slope and intercept to the plot
    print(p)
    plt.text(x[0], y[0], f"{p}", fontsize=12, color="red", backgroundcolor="white")
    return z


if __name__ == "__main__":
    # data = "data\SPG806_20240804_measure_enable_response_D6_A4_2024-08-04 18-59-58.mat"
    data = "data\SPG806_20240806_measure_enable_response_D6_A4_B1_2024-08-06 17-30-39.mat"
# C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\data\SPG806_20240806_measure_enable_response_D6_A4_B3_2024-08-06 16-22-06.mat
    enable_current = 0.0
    # enable_current_relation(data, enable_current)
    data_dict = sio.loadmat(data)
    find_peak(data_dict)

    for key in data_dict.keys():
        if isinstance(data_dict[key], np.ndarray):
            print(f"key: {key}, shape: {data_dict[key].shape}")
