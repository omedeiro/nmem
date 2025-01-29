
import matplotlib.pyplot as plt

from nmem.analysis.analysis import plot_read_sweep_array, import_directory


# def plot_read_temp_sweep_C3():
#     fig, axs = plt.subplots(2, 2, figsize=(12, 6))

#     plot_write_sweep(axs[0, 0], "write_current_sweep_C3_2")
#     plot_write_sweep(axs[0, 1], "write_current_sweep_C3_3")
#     plot_write_sweep(axs[1, 0], "write_current_sweep_C3_4")
#     plot_write_sweep(axs[1, 1], "write_current_sweep_C3")
#     axs[1, 1].legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
#     fig.subplots_adjust(hspace=0.5, wspace=0.3)


if __name__ == "__main__":
    # plot_write_sweep("write_current_sweep_B2_0")
    # plot_write_sweep("write_current_sweep_B2_1")
    # plot_write_sweep("write_current_sweep_B2_2")

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, import_directory("write_current_sweep_A2"), "bit_error_rate", "write_current")
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, import_directory("write_current_sweep_C2"), "bit_error_rate", "write_current")
    plt.show()


