import matplotlib.pyplot as plt

from nmem.analysis.memory_data import (
    axis_max,
    axis_min,
    datasets,
    labels,
    metrics,
    normalizers,
    styles,
    units,
)
from nmem.analysis.spider_plots import (
    plot_radar,
)
from nmem.analysis.styles import apply_global_style

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    # --- Call the updated plot function ---
    plot_radar(
        metrics,
        units,
        axis_min,
        axis_max,
        normalizers,
        datasets,
        labels,
        styles,
    )

    if save_dir:
        plt.savefig(
            f"{save_dir}/compare_metrics_spider.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
