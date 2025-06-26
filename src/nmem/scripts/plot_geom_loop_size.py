import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator

from nmem.analysis.core_analysis import analyze_geom_loop_size
from nmem.analysis.data_import import import_geom_loop_size_data
from nmem.analysis.styles import apply_global_style

# Apply global plot styling
apply_global_style()



def main(data_dir="../data/loop_size_sweep", save_dir=None):
    data, loop_sizes = import_geom_loop_size_data(data_dir)
    vch_list, ber_est_list, err_list, best_ber = analyze_geom_loop_size(
        data, loop_sizes
    )
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
    # First plot: Vch vs. ber_est
    ax = axs[0]
    for i in range(len(data)):
        ax.plot(
            vch_list[i], ber_est_list[i], label=f"$w_{{5}}$ = {loop_sizes[i]:.1f} µm"
        )
    ax.set_yscale("log")
    ax.set_xlabel("channel voltage [mV]")
    ax.set_ylabel("estimated BER")
    ax.legend(loc="lower left", labelspacing=0.1, handlelength=1.5, fontsize=7)
    # Second plot: best BER vs loop size
    ax = axs[1]
    ax.plot(loop_sizes, best_ber, "-o")
    ax.set_yscale("log")
    ax.set_xlabel("loop size [µm]")
    ax.set_ylabel("minimum BER")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.2)
    axs[2].axis("off")
    if save_dir:
        plt.savefig(
            f"{save_dir}/geom_loop_size.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
