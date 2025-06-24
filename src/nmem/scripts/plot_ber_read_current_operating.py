import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_read_current_operating


def main(save_dir=None):
    dict_list = import_directory(
        "../data/ber_sweep_read_current/write_current/write_current_sweep_C3"
    )
    plot_read_current_operating(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_operating.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
