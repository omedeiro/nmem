
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from nmem.analysis.analysis import (
    plot_read_sweep_array,
)
from nmem.calculations.analytical_model import create_data_dict
from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 6
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.markeredgewidth"] = 0.5

plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1

CURRENT_CELL = "C1"


# def plot_data_delay_manu_dev(
#     axs: List[Axes],
#     data_dict_keyd: dict,
# ) -> List[Axes]:

#     cmap = plt.get_cmap("RdBu").reversed()
#     colors = cmap(np.linspace(0, 1, 8))

#     INDEX = 14

#     ax = axs[0]
#     data_dict = data_dict_keyd[0]
#     x, y = get_trace_data(data_dict, "trace_chan_in", INDEX)
#     # y = np.mean(y)
    
#     plot_voltage_trace(ax, x, y, color=colors[0])

#     ax = axs[1]
#     data_dict = data_dict_keyd[1]
#     x, y = get_trace_data(data_dict, "trace_chan_in", INDEX)
#     plot_voltage_trace(ax, x, y, color=colors[-1])

#     for idx, ax in enumerate(axs[2:]):
#         data_dict = data_dict_keyd[idx % len(data_dict_keyd)]
#         x, y = get_trace_data(data_dict, "trace_chan_in", INDEX)
#         plot_voltage_trace(ax, x, y, color=colors[1])
#         plot_threshold(ax, 4, 5, 400)
#         plot_threshold(ax, 9, 10, 400)
#         ax.set_ylim([-150, 1100])
#     ax.set_xlim([0, 10])
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.xaxis.set_major_formatter(lambda x, _: f"{x:.1f}")
#     fig = plt.gcf()
#     fig.subplots_adjust(hspace=0.0, bottom=0.05, top=0.95)
#     fig.supxlabel("Time ($\mu$s)", x=0.5, y=-0.03)
#     fig.supylabel("Voltage (mV)", x=0.05, y=0.5)
#     return axs


# def manuscript_figure(save: bool = False) -> None:
#     fig = plt.figure(figsize=(9.5, 3.5))

#     subfigs = fig.subfigures(1, 3, wspace=-0.3, width_ratios=[0.5, 1, 1])

#     axs = subfigs[0].subplots(6, 1, sharex=True, sharey=False)
#     plot_data_delay_manu_dev(axs, inverse_compare_dict)
#     subfigs[0].supxlabel("Time ($\mu$s)", x=0.5, y=-0.01)
#     subfigs[0].supylabel("Voltage (mV)", x=1.01, y=0.5, rotation=270)
#     subfigs[0].subplots_adjust(hspace=0.0, bottom=0.05, top=0.95)

#     axsslice = subfigs[1].subplots(3, 1, subplot_kw={"projection": "3d"})
#     plot_read_sweep_array_3d(axsslice[0], enable_read_290_dict)
#     plot_read_sweep_array_3d(axsslice[1], enable_read_300_dict)
#     plot_read_sweep_array_3d(axsslice[2], enable_read_310_dict)
#     subfigs[1].subplots_adjust(hspace=-0.6, bottom=-0.2, top=1.20, left=0.1, right=1.1)

#     axsstack = subfigs[2].subplots(3, 1, sharex=True)
#     plot_stack(
#         axsstack,
#         [analytical_data_dict, analytical_data_dict, analytical_data_dict],
#         [-30, 0, 30],
#     )
#     subfigs[2].subplots_adjust(hspace=0.0, bottom=0.06, top=0.90, left=-0.2, right=0.9)
#     subfigs[2].supxlabel("$I_{{CH}}$ ($\mu$A)", x=0.36, y=-0.02)
#     subfigs[2].supylabel("$I_{{R}}$ ($\mu$A)", x=0.48, y=0.5)

#     caxis = subfigs[2].add_axes([0.3, 0.92, 0.1, 0.02], frame_on=True)
#     subfigs[2].colorbar(
#         axsstack[0].collections[-1], cax=caxis, orientation="horizontal", pad=0.1
#     )
#     caxis.tick_params(
#         labeltop=True, labelbottom=False, bottom=False, top=True, direction="out"
#     )
#     subfigs[2].supxlabel("$I_{{CH}}$ ($\mu$A)")
#     subfigs[2].supylabel("$I_{{R}}$ ($\mu$A)")
#     fig.patch.set_visible(False)
#     if save:
#         plt.savefig("trace_waterfall_fit_combined.pdf", bbox_inches="tight")
#     plt.show()
#     return


# def plot_trace_only() -> None:
#     fig, axs = plt.subplots(6, 1, figsize=(6.6, 3.54))
#     axs = plot_data_delay_manu_dev(axs, INVERSE_COMPARE_DICT)
#     fig.subplots_adjust(hspace=0.0, bottom=0.05, top=0.95)
#     fig.supxlabel("Time ($\mu$s)", x=0.5, y=-0.03)
#     fig.supylabel("Voltage (mV)", x=0.95, y=0.5, rotation=270)
#     plt.savefig("trace_only.png", bbox_inches="tight", dpi=300)
#     plt.show()


if __name__ == "__main__":
    INVERSE_COMPARE_DICT = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 16-04-36.mat"
        ),
    }
    write_dict = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
    }

    data_dict = {
        0: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-10-41.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-17-47.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-11-46.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-18-32.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-25-36.mat"
        ),
    }

    enable_read_290_dict_full = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-45-11.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-53-18.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-08-51.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-19-03.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-10-30.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-26-55.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-33-48.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-40-47.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-49-39.mat"
        ),
        10: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-59-27.mat"
        ),
        11: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        12: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-17-53.mat"
        ),
        13: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
        14: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-32-29.mat"
        ),
        15: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-40-00.mat"
        ),
        16: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-53-35.mat"
        ),
        17: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-02-47.mat"
        ),
    }

    enable_read_290_dict = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-45-11.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-53-18.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-08-51.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-19-03.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-33-48.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-49-39.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-40-00.mat"
        ),
        10: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-02-47.mat"
        ),
    }

    enable_read_290_dict_short = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
    }

    enable_read_300_dict = {
        0: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-54-06.mat"
        ),
        1: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-06-21.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-44-04.mat"
        ),
        3: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-16-22.mat"
        ),
        4: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-23-31.mat"
        ),
        5: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-30-28.mat"
        ),
        6: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-39-15.mat"
        ),
        7: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-47-05.mat"
        ),
        8: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-54-14.mat"
        ),
        9: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 16-04-36.mat"
        ),
        10: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-29-17.mat"
        ),
    }

    enable_read_310_dict = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-37-34.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-30-17.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-21-01.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-13-38.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-15-11.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-27-11.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-34-04.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-44-33.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-56-20.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-05-30.mat"
        ),
        10: sio.loadmat(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 09-17-27.mat"
        ),
    }

    enable_read_310_C4_dict = {
        0: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 13-49-54.mat"
        ),
        1: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-03-11.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-10-42.mat"
        ),
        3: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-17-31.mat"
        ),
        4: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-24-30.mat"
        ),
        5: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-43-26.mat"
        ),
        6: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-45-10.mat"
        ),
        7: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-52-08.mat"
        ),
        8: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-06-17.mat"
        ),
        9: sio.loadmat(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-13-23.mat"
        ),
    }

    inverse_compare_dict = {
        0: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
    }

    fitting_dict = {
        -30: {
            0: {"fit_start": 0, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 0},
            2: {"fit_start": 0, "fit_stop": 0},
            3: {"fit_start": 0, "fit_stop": 0},
        },
        0: {
            0: {"fit_start": 1, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 2},
            2: {"fit_start": 0, "fit_stop": 1},
            3: {"fit_start": 0, "fit_stop": 2},
        },
        30: {
            0: {"fit_start": 1, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 2},
            2: {"fit_start": 0, "fit_stop": 5},
            3: {"fit_start": 0, "fit_stop": 1},
        },
    }

    current_cell = "C1"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["y_intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.213
    ALPHA = 0.563
    PERSISTENT_CURRENT = 30.0
    MAX_CRITICAL_CURRENT = 860e-6  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.573
    IREAD = 630
    N = 200

    enable_read_currents = np.linspace(0, 400, N)
    read_currents = np.linspace(400, 1050, N)

    analytical_data_dict = create_data_dict(
        enable_read_currents,
        read_currents,
        WIDTH_LEFT,
        WIDTH_RIGHT,
        ALPHA,
        IRETRAP_ENABLE,
        MAX_CRITICAL_CURRENT,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
        PERSISTENT_CURRENT,
    )

    # manuscript_figure()
    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, enable_read_310_C4_dict, "bit_error_rate", "enable_read_current"
    )
    plt.show()

