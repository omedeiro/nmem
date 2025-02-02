import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
    plot_state_currents,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_state_currents(ax, data_list)
    plt.show()

    # fig, ax = plt.subplots(figsize=(6, 4))
    # plot_state_separation(ax, data_list)
    # plt.show()

    # fig, ax = plt.subplot_mosaic(
    #     [["A", "B", "C", "D"], ["E", "E", "E", "E"]],
    #     figsize=(16, 9),
    #     tight_layout=True,
    #     sharex=False,
    #     sharey=False,
    # )
    # plot_enable_write_sweep_grid(ax, data_list)
    # plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "enable_write_current")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    plt.show()
