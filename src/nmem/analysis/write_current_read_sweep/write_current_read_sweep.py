import matplotlib.pyplot as plt
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data_list = import_directory("data")
    data_list = data_list[::-1]
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "write_current")
    plt.show()

    data_list2 = import_directory("data2")
    data_list2 = data_list2[::-1]
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list2, "bit_error_rate", "write_current")
    plt.show()
