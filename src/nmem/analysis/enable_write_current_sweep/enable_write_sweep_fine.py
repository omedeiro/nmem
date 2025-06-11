import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import plot_enable_write_sweep_fine


def main(data_dir="data2", save_fig=False, output_path="enable_write_sweep_fine.pdf"):
    """
    Main function to plot fine enable write sweep.
    """
    data_list2 = import_directory(data_dir)
    plot_enable_write_sweep_fine(data_list2, save_fig=save_fig, output_path=output_path)
    plt.show()


if __name__ == "__main__":
    main()
