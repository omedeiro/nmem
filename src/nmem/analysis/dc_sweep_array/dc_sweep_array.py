
import matplotlib.pyplot as plt

from nmem.analysis.currents import extract_ic_vs_ih_data
from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_ic_vs_ih_array
from nmem.analysis.styles import set_plot_style

set_plot_style()


def main(data_dir=".", save_fig=False, output_path="ic_vs_ih_array.png"):
    """
    Main function to extract data and plot Ic vs Ih array.
    """
    set_plot_style()
    data = import_directory(data_dir)[0]
    heater_currents, avg_current, ystd, cell_names = extract_ic_vs_ih_data(data)
    fig, ax = plot_ic_vs_ih_array(
        heater_currents, avg_current, ystd, cell_names, save_fig, output_path
    )
    plt.show()


if __name__ == "__main__":
    main()
