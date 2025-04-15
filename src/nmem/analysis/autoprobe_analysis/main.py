from nmem.analysis.autoprobe_analysis.data import load_autoprobe_data, build_resistance_map, normalize_row_by_squares
from nmem.analysis.autoprobe_analysis.plot import plot_resistance_map, plot_die_resistance_map
from nmem.analysis.autoprobe_analysis.utils import build_device_lookup
import matplotlib.pyplot as plt


def main(data_path="autoprobe_parsed.mat"):
    df = load_autoprobe_data(data_path)
    print(df)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_resistance_map(ax, df)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_die_resistance_map(ax, df, "E2", annotate=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")

    main()
