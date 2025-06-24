from matplotlib import pyplot as plt
from nmem.analysis.core_analysis import analyze_prbs_errors
from nmem.analysis.data_import import import_directory
from nmem.analysis.trace_plots import plot_probe_station_prbs

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(data_dir="../data/ber_prbs_probe_station", trim=4500, save_dir=None):
    data_list = import_directory(data_dir)
    total_error, W1R0_error, W0R1_error, error_locs = analyze_prbs_errors(
        data_list, trim=trim
    )
    fig, ax = plot_probe_station_prbs(
        data_list,
        trim=trim,
        error_locs=error_locs,
    )
    logger.info(f"Total errors: {total_error}, W1R0: {W1R0_error}, W0R1: {W0R1_error}")

    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_prbs_probe_station_trace.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
