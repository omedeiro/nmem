import matplotlib.pyplot as plt

from nmem.analysis.data_import import load_autoprobe_data
from nmem.analysis.matrix_plots import plot_combined_histogram_and_die_maps
from nmem.analysis.utils import summarize_die_yield


def main(save_dir=None):
    df = load_autoprobe_data("../data/wafer_autoprobe_resistance/autoprobe_parsed.mat")
    wafer_rows = ["1", "4", "6", "7"]
    summary_df, row_stats_df = summarize_die_yield(
        df, wafer_rows, min_kohm=0.1, max_kohm=500000
    )

    limit_dict = {
        "1": [20, 100],
        "4": [20, 100],
        "6": [900, 1100],
        "7": [20, 100],
    }
    plot_combined_histogram_and_die_maps(df, wafer_rows, limit_dict)

    if save_dir:
        plt.savefig(
            f"{save_dir}/autoprobe_resistance_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()

    # print(summary_df)  # Die-level stats
    # print(row_stats_df.head())  # Row-level summary


if __name__ == "__main__":
    main()
