import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
)

if __name__ == "__main__":
    data_list2 = import_directory("data2")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enable_write_sweep_multiple(ax, data_list2)
    ax.set_xlim([260, 310])
    plt.savefig("enable_write_sweep_fine.pdf", bbox_inches="tight")
    plt.show()
