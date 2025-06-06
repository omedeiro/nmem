import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import plot_enable_write_sweep_multiple

if __name__ == "__main__":
    data_list2 = import_directory("data2")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enable_write_sweep_multiple(ax, data_list2)
    ax.set_xlim([260, 310])
    save_fig = False
    if save_fig:
        plt.savefig("enable_write_sweep_fine.pdf", bbox_inches="tight")
    plt.show()
