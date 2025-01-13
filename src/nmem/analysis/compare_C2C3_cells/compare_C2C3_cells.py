import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.analysis.analysis import get_fitting_points, build_array
from nmem.measurement.functions import plot_fitting


def calculate_channel_temperature(data_dict: dict) -> np.ndarray:

    temp_critical: float = data_dict.get("critical_temperature", 12.3)
    temp_substrate: float = data_dict.get("substrate_temperature", 1.3)

    ih = data_dict.get("x")[0][:, 0]
    ih_max = data_dict.get("x_intercept", 500e-6)

    return _calculate_channel_temperature(temp_critical, temp_substrate, ih, ih_max)


def _calculate_channel_temperature(
    temp_critical: float, temp_substrate: float, ih: np.ndarray, ih_max: float
) -> np.ndarray:
    N = 2.0
    temp_channel = (
        (temp_critical**4 - temp_substrate**4) * (ih / ih_max) ** N
        + (temp_substrate**4)
    ) ** 0.25
    return temp_channel


def calculate_critical_current(data_dict: dict) -> np.ndarray:
    temp_critical = data_dict.get("critical_temperature", 12.3)
    temp_substrate = data_dict.get("substrate_temperature", 1.3)
    temp_channel = calculate_channel_temperature(data_dict)

    ic_max = data_dict.get("max_critical_current", 900e-6)
    return _calculate_critical_current(
        temp_critical, temp_substrate, temp_channel, ic_max
    )


def _calculate_critical_current(
    temp_critical: float, temp_substrate: float, temp_channel: np.ndarray, ic_max: float
) -> np.ndarray:
    ic_zero = ic_max / (1 + (temp_substrate / temp_critical) ** 3) ** 2.1

    ic = ic_zero * (1 + (temp_channel / temp_critical) ** 3) ** 2.1
    return ic


def plot_channel_temperature(ax: plt.Axes, data_dict: dict) -> np.ndarray:
    channel_temp = calculate_channel_temperature(data_dict)
    heater_current = data_dict.get("x")[0][:, 0] * 1e6
    ax.plot(heater_current, channel_temp, label="C2", linestyle="-")
    ax.set_xlabel("Heater Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")

    return ax


if __name__ == "__main__":
    data_dict = sio.loadmat(
        r"SPG806_20241220_nMem_measure_enable_response_D6_A4_C2_2024-12-20 13-28-02.mat"
    )
    data_dict2 = sio.loadmat(
        r"SPG806_20241220_nMem_measure_enable_response_D6_A4_C3_2024-12-20 17-28-57.mat"
    )

    split_idx = 10

    fig, axs = plt.subplots()
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    axs.plot(xfit, yfit, label="C2", linestyle="-")
    split_idx = 7
    plot_fitting(axs, xfit[split_idx + 1 :], yfit[split_idx + 1 :])

    split_idx = 10
    x2, y2, ztotal2 = build_array(data_dict2, "total_switches_norm")

    xfit, yfit = get_fitting_points(x2, y2, ztotal2)
    axs.plot(xfit, yfit, label="C3", linestyle="-")
    axs.legend()
    axs.set_ylim([0, 1000])
    axs.set_xlim([0, 500])

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    plot_fitting(
        axs[0], xfit[split_idx + 1 :], yfit[split_idx + 1 :], label="C3", linestyle="-"
    )
    axs[0].set_ylim([0, 1000])
    axs[0].set_xlim([0, 500])
    axs[0].plot(xfit, yfit, label="C2", linestyle="-")

    plot_fitting(axs[1], xfit[:split_idx], yfit[:split_idx], label="C3", linestyle="-")
    axs[1].plot(xfit, yfit, label="C2", linestyle="-")
    axs[1].set_ylim([0, 1000])
    axs[1].set_xlim([0, 500])


    # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    # axs[0].plot(xfit, yfit, label="C2", linestyle="-")

    fig, ax = plt.subplots()
    plot_channel_temperature(ax, data_dict)
    plot_channel_temperature(ax, data_dict2)
