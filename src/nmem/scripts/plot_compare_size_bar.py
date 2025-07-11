import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.bar_extrusion import (
    plot_extruded_bar,
)
from nmem.analysis.memory_data import (
    cap_colors,
    cap_labels,
    cap_vals,
    den_colors,
    den_labels,
    den_vals,
    tech_cap,
    tech_den,
)
from nmem.analysis.styles import set_inter_font, set_pres_style

set_inter_font()
set_pres_style()


def main(save_dir=None):
    """
    Generate memory scaling comparison plots showing capacity and density.
    """
    # ---------------------- Create Figure ----------------------
    fig, axs = plt.subplots(2, 1, figsize=(8, 9), sharex=False)
    fig.suptitle(
        "Superconducting vs. Semiconducting Memory Scaling Comparison",
        weight="bold",
        y=0.96,
    )

    # Plot 1: Capacity
    plot_extruded_bar(
        tech_cap,
        cap_vals,
        cap_colors,
        orientation="h",
        depth=0.15,
        xlabel="Memory Capacity (log scale, bits/chip)",
        xticks=[np.log10(2**10), np.log10(2**20), np.log10(2**30), np.log10(2**40)],
        xticklabels=[
            "1 kb",
            "1 Mb",
            "1 Gb",
            "1 Tb",
        ],
        bar_labels=cap_labels,
        ax=axs[1],
        annotation_offset=-0.05,
    )

    # Plot 2: Density
    plot_extruded_bar(
        tech_den,
        den_vals,
        den_colors,
        orientation="h",
        depth=0.15,
        xlabel="Functional Density (log scale, bits/cm²)",
        xticks=[np.log10(1e6), np.log10(1e9), np.log10(1e12)],
        xticklabels=["1 Mb/cm²", "1 Gb/cm²", "1 Tb/cm²"],
        bar_labels=den_labels,
        annotation_offset=-0.05,
        ax=axs[0],
    )

    if save_dir:
        plt.savefig(f"{save_dir}/compare_size_bar.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
