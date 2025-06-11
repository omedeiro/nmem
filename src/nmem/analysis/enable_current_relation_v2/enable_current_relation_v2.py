import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_column,
    plot_full_grid,
    plot_grid,
    plot_row,
)
from nmem.analysis.styles import set_plot_style


def main(
    data_dir="data",
    save_full_grid=False,
    full_grid_path="enable_current_relation_full_grid.pdf",
    save_grid=False,
    grid_path="enable_current_relation_grid.png",
):
    """
    Main function to plot enable current relation grids and columns/rows.
    """
    set_plot_style()
    dict_list = import_directory(data_dir)

    fig1, axs1 = plt.subplots(5, 5, figsize=(6, 6), sharex=True, sharey=True)
    plot_full_grid(axs1, dict_list)
    if save_full_grid:
        fig1.savefig(full_grid_path, bbox_inches="tight")
    plt.show()

    fig2, axs2 = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_column(axs2, dict_list)
    axs2[0].set_xlabel("Enable Current ($\\mu$A)")
    axs2[0].set_ylabel("Critical Current ($\\mu$A)")
    plt.show()

    fig3, axs3 = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_row(axs3, dict_list)
    axs3[0].set_xlabel("Enable Current ($\\mu$A)")
    axs3[0].set_ylabel("Critical Current ($\\mu$A)")
    plt.show()

    fig4, axs4 = plt.subplots(
        4, 4, figsize=(180 / 25.4, 180 / 25.4), sharex=True, sharey=True
    )
    plot_grid(axs4, dict_list)
    if save_grid:
        fig4.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
