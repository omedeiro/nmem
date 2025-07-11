import matplotlib.pyplot as plt

from nmem.analysis.data_import import (
    import_write_sweep_formatted,
    import_write_sweep_formatted_markers,
)
from nmem.analysis.sweep_plots import (
    plot_write_sweep_formatted,
    plot_write_sweep_formatted_markers,
)


def main(save_dir=None):
    innerb = [
        ["C", "D"],
    ]
    fig, axs = plt.subplot_mosaic(
        innerb,
        figsize=(6, 2),
    )
    dict_list = import_write_sweep_formatted()
    plot_write_sweep_formatted(axs["C"], dict_list)
    data_dict = import_write_sweep_formatted_markers(dict_list)
    plot_write_sweep_formatted_markers(axs["D"], data_dict)
    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.1,
        hspace=0.4,
        wspace=0.7,
    )

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_write_current_sweep_enable_margin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
