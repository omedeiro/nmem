import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_current_relation,
)

plt.rcParams["figure.figsize"] = [5.7, 5]
plt.rcParams["font.size"] = 16


if __name__ == "__main__":
    dict_list = import_directory("data")
    fig, ax = plt.subplots()
    plot_enable_current_relation(ax, dict_list)
    plt.show()
