import os

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_enable_sweep


def main():
    fig, axs = plt.subplot_mosaic("BC", figsize=(180 / 25.4, 90 / 25.4))
    dict_list = import_directory(
        os.path.join(os.path.dirname(__file__), "data")
    )
    sort_dict_list = sorted(
        dict_list, key=lambda x: x.get("write_current").flatten()[0]
    )
    plot_enable_sweep(
        axs["B"],
        sort_dict_list,
        range=slice(0, len(sort_dict_list) // 2),
        add_errorbar=False,
        add_colorbar=False,
    )
    plot_enable_sweep(
        axs["C"],
        sort_dict_list,
        range=slice(len(sort_dict_list) // 2, len(sort_dict_list)),
        add_errorbar=False,
        add_colorbar=True,
    )
    # plt.savefig("sup_full_param_sweeps.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
