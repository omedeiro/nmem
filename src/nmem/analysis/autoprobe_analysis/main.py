import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.plotting import set_plot_style, plot_combined_histogram_and_die_maps
from nmem.analysis.data_import import load_autoprobe_data

set_plot_style()

if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")
    wafer_rows = ["1", "4", "6", "7"]
    limit_dict = {
        "1": [20, 100],
        "4": [20, 100],
        "6": [900, 1100],
        "7": [20, 100],
    }
    plot_combined_histogram_and_die_maps(df, wafer_rows, limit_dict)
