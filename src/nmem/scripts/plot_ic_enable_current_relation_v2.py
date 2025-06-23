import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import set_plot_style
from nmem.analysis.sweep_plots import (
    plot_column,
    plot_full_grid,
    plot_grid,
    plot_row,
)


def main(
    data_dir="../data/enable_current_relation/data3",
    save_full_grid=False,
    full_grid_path="enable_current_relation_full_grid.pdf",
    save_grid=False,
    grid_path="enable_current_relation_grid.png",
    save_dir=None,
):
    """
    Main function to plot enable current relation grids and columns/rows.

    Args:
        save_dir: If provided, saves figures to this directory instead of showing them
    """
    import os

    set_plot_style()
    dict_list = import_directory(data_dir)

    # Figure 1: Full grid
    fig1, axs1 = plt.subplots(5, 5, figsize=(6, 6), sharex=True, sharey=True)
    plot_full_grid(axs1, dict_list)
    if save_dir:
        fig1.savefig(
            os.path.join(save_dir, "plot_ic_enable_current_relation_v2_full_grid.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1)
    elif save_full_grid:
        fig1.savefig(full_grid_path, bbox_inches="tight")
        plt.show()
    else:
        plt.show()

    # Figure 2: Columns
    fig2, axs2 = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_column(axs2, dict_list)
    axs2[0].set_xlabel("Enable Current ($\\mu$A)")
    axs2[0].set_ylabel("Critical Current ($\\mu$A)")
    if save_dir:
        fig2.savefig(
            os.path.join(save_dir, "plot_ic_enable_current_relation_v2_columns.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)
    else:
        plt.show()

    # Figure 3: Rows
    fig3, axs3 = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_row(axs3, dict_list)
    axs3[0].set_xlabel("Enable Current ($\\mu$A)")
    axs3[0].set_ylabel("Critical Current ($\\mu$A)")
    if save_dir:
        fig3.savefig(
            os.path.join(save_dir, "plot_ic_enable_current_relation_v2_rows.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig3)
    else:
        plt.show()

    # Figure 4: Grid
    fig4, axs4 = plt.subplots(
        4, 4, figsize=(180 / 25.4, 180 / 25.4), sharex=True, sharey=True
    )
    plot_grid(axs4, dict_list)
    if save_dir:
        fig4.savefig(
            os.path.join(save_dir, "plot_ic_enable_current_relation_v2_grid.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig4)
    elif save_grid:
        fig4.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()
