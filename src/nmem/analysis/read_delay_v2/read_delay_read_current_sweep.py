import matplotlib.pyplot as plt

from nmem.analysis.analysis import import_directory, plot_read_delay

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_read_delay(ax, dict_list)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    plt.show()
