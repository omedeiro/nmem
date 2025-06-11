import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    get_fitting_points,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_c2c3_comparison,
    plot_c2c3_subplots,
)
from nmem.analysis.utils import build_array



def main(
    data_dir="data",
    save_fig=False,
    output_path="enable_current_relation_compare_C2C3.png",
):
    """
    Main function to compare C2 and C3 cells and plot results.
    """
    data_list = import_directory(data_dir)
    data_dict = data_list[0]
    data_dict2 = data_list[1]
    split_idx = 10
    # First plot: single axis comparison
    fig1, ax1 = plt.subplots()
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    plot_c2c3_comparison(ax1, xfit, yfit, split_idx, label_c2="C2", label_c3="C3")
    # Second plot: subplots for C2 and C3
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 10))
    x2, y2, ztotal2 = build_array(data_dict2, "total_switches_norm")
    xfit2, yfit2 = get_fitting_points(x2, y2, ztotal2)
    plot_c2c3_subplots(axs2, xfit2, yfit2, split_idx, label_c2="C2", label_c3="C3")
    if save_fig:
        fig2.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
