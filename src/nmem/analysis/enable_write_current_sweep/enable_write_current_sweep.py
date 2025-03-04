import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
)

if __name__ == "__main__":
    # Import
    dict_list = import_directory("data")

    # Plot
    fig, ax = plt.subplots()
    ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    save=False
    if save:
        fig.savefig("enable_write_sweep.pdf", bbox_inches="tight")
    plt.show()