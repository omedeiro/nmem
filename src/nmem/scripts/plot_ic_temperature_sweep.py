import matplotlib.pyplot as plt

from nmem.analysis.currents import extract_ic_vs_ih_data
from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_ic_vs_ih_array

apply_global_style()


def main(data_dir="../data/dc_sweep_array", save_dir=None):
    """
    Main function to extract data and plot Ic vs Ih array.
    """
    data = import_directory(data_dir)[0]
    heater_currents, avg_current, ystd, cell_names = extract_ic_vs_ih_data(data)
    fig, ax = plot_ic_vs_ih_array(heater_currents, avg_current, ystd, cell_names)

    if save_dir:
        fig.savefig(
            f"{save_dir}/ic_vs_ih_array.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
