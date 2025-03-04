import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
)

if __name__ == "__main__":
    # Import
    data_list2 = import_directory("data2")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enable_write_sweep_multiple(ax, data_list2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    save = False
    if save:
        fig.savefig("enable_write_sweep_fine.pdf", bbox_inches="tight")
    plt.show()
