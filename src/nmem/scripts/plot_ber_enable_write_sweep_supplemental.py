import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_enable_sweep


def main(save_dir=None):
    fig, axs = plt.subplot_mosaic("BC", figsize=(180 / 25.4, 90 / 25.4))
    dict_list = import_directory("../data/ber_sweep_enable_write_current/data1")
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

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_supplemental.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
