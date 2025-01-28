import matplotlib.pyplot as plt

from nmem.analysis.analysis import import_directory, plot_read_delay

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 12


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_read_delay(ax, dict_list)
    plt.show()