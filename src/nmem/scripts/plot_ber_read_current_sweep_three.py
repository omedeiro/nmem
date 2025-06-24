import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_read_current_sweep_three_data
from nmem.analysis.sweep_plots import plot_read_current_sweep_three


def main(save_dir=None):
    dict_list = import_read_current_sweep_three_data()
    plot_read_current_sweep_three(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_three.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
