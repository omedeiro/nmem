from nmem.analysis.autoprobe_analysis.data import (
    load_autoprobe_data,
    build_resistance_map,
    normalize_row_by_squares,
)
from nmem.analysis.autoprobe_analysis.plot import (
    plot_resistance_map,
    plot_die_resistance_map,
    plot_die_row,
    scatter_die_row_resistance,
    scatter_die_resistance,
)
import matplotlib.pyplot as plt


def main(data_path="autoprobe_parsed.mat"):
    df = load_autoprobe_data(data_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_resistance_map(ax, df)
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_die_resistance_map(ax, df, "G4", annotate=True)
    plt.show()

    fig, axs = plt.subplots(1, 7, figsize=(12, 8))
    plot_die_row(axs, df, 1, annotate=True)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax = scatter_die_row_resistance(ax, df, 7, logscale=False)
    plt.show()


if __name__ == "__main__":
    df = load_autoprobe_data("autoprobe_parsed.mat")

    main()
