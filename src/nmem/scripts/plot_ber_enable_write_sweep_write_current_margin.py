import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_write_current_enable_sweep_margin

# Apply global plot styling
apply_global_style()



def main(data_dir="../data/ber_sweep_enable_write_current/data1", save_dir=None):
    """
    Main function to plot write current enable sweep margin.
    """
    inner = [
        ["A", "B"],
    ]
    dict_list = import_directory(data_dir)
    plot_write_current_enable_sweep_margin(
        dict_list,
        inner,
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
