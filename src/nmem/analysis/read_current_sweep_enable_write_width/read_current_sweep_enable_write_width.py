import matplotlib.pyplot as plt

from nmem.analysis.analysis import import_directory, plot_read_sweep_array


if __name__ == "__main__":
    # Import
    dict_list = import_directory("data")


    # Plot
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_write_width")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Write Width [pts]",
    )
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1])
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    plt.show()
