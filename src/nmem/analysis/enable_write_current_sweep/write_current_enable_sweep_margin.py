import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_write_current_enable_sweep_margin


def main(
    data_dir="data", save_fig=False, output_path="write_current_enable_sweep_margin.pdf"
):
    """
    Main function to plot write current enable sweep margin.
    """
    inner = [
        ["A", "B"],
    ]
    dict_list = import_directory(data_dir)
    plot_write_current_enable_sweep_margin(
        dict_list, inner, save_fig=save_fig, output_path=output_path
    )
    plt.show()


if __name__ == "__main__":
    main()
