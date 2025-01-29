import matplotlib.pyplot as plt
from nmem.analysis.analysis import import_directory, plot_read_sweep_array

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_current")
    ax.set_ylim(0, 0.5)
