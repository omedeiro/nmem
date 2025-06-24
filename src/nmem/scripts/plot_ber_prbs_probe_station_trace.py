
from nmem.analysis.core_analysis import analyze_prbs_errors
from nmem.analysis.data_import import import_directory
from nmem.analysis.trace_plots import plot_probe_station_prbs

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main(
    data_dir="../data/ber_prbs_probe_station", trim=4500, save_fig=False, output_path="probe_station_prbs.pdf"
):
    data_list = import_directory(data_dir)
    total_error, W1R0_error, W0R1_error, error_locs = analyze_prbs_errors(
        data_list, trim=trim
    )
    fig, ax = plot_probe_station_prbs(
        data_list,
        trim=trim,
        error_locs=error_locs,
        save_fig=save_fig,
        output_path=output_path,
    )
    logger.info(f"Total errors: {total_error}, W1R0: {W1R0_error}, W0R1: {W0R1_error}")


if __name__ == "__main__":
    main()
