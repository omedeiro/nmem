import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.pyplot import Axes
from nmem.analysis.analysis import (
    plot_read_sweep,
    plot_read_sweep_array,
)
from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


CURRENT_CELL = "C1"


def plot_read_sweep_multiple(ax: Axes, data_dict: dict) -> Axes:
    for key in data_dict.keys():
        ax = plot_read_sweep(ax, data_dict[key], "bit_error_rate", "write_current")

    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


if __name__ == "__main__":
    current_cell = "C4"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.213
    ALPHA = 0.563

    MAX_CRITICAL_CURRENT = 860e-6  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.573
    IREAD = 630
    N = 200

    write_read_sweep_C4_dict = {
        0: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-48-03.mat"
        ),
        1: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-39-50.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-28-13.mat"
        ),
        3: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-09-02.mat"
        ),
        4: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-17-33.mat"
        ),
        5: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-56-41.mat"
        ),
        6: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-11-47.mat"
        ),
        7: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-19-41.mat"
        ),
        8: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-28-51.mat"
        ),
        9: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-36-42.mat"
        ),
        10: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-18-14.mat"
        ),
        11: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-26-34.mat"
        ),
        12: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-34-47.mat"
        ),
    }

    write_read_sweep_C4_dict_min_pulsewidth = {
        0: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-51-16.mat"
        ),
        1: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-47-12.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-43-05.mat"
        ),
        3: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-36-19.mat"
        ),
        4: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-31-59.mat"
        ),
        5: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-27-18.mat"
        ),
        6: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-05-17.mat"
        ),
        7: sio.loadmat(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-00-10.mat"
        ),
    }

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, write_read_sweep_C4_dict_min_pulsewidth, "bit_error_rate", "write_current"
    )
