import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.htron_plotting import plot_enable_current_relation


def main():
    dict_list = import_directory("../data/enable_current_relation/data1")
    fig, ax = plt.subplots()
    plot_enable_current_relation(ax, dict_list)

    ax.set_xlabel("Enable Current ($\mu$A)")
    ax.set_ylabel("Critical Current ($\mu$A)")
    ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")

    save = False
    if save:
        plt.savefig("enable_current_relation.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
