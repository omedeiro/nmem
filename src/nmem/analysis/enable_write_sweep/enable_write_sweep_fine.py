import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
)

if __name__ == "__main__":
    data_list2 = import_directory("data2")
    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_list2)
    plt.show()