import os
from nmem.analysis.alignment_plots import (
    plot_alignment_offset_hist,
    plot_alignment_stats,
)
from nmem.analysis.core_analysis import analyze_alignment_stats
from nmem.analysis.data_import import import_elionix_log


def main(
    log_path="../data/elionix_schedule/New schedule.log",
    save=False,
    save_fig=False,
    analysis_path="alignment_analysis.pdf",
    offset_hist_path="alignment_offsets_histogram.pdf",
    save_dir=None,
):
    """
    Main function to import, analyze, and plot Elionix alignment log data.

    Args:
        save_dir: If provided, saves figures to this directory instead of using individual save parameters
    """

    df_z, df_rot_valid, dx_nm, dy_nm, delta_table = import_elionix_log(log_path)
    z_mean, z_std, r_mean, r_std = analyze_alignment_stats(
        df_z, df_rot_valid, dx_nm, dy_nm
    )

    # Plot 1: Alignment statistics
    if save_dir:
        analysis_save_path = os.path.join(save_dir, "elionix_log_alignment_stats.png")
        plot_alignment_stats(
            df_z,
            df_rot_valid,
            dx_nm,
            dy_nm,
            z_mean,
            z_std,
            r_mean,
            r_std,
            save=True,
            output_path=analysis_save_path,
        )
    else:
        plot_alignment_stats(
            df_z,
            df_rot_valid,
            dx_nm,
            dy_nm,
            z_mean,
            z_std,
            r_mean,
            r_std,
            save=save,
            output_path=analysis_path,
        )

    # Plot 2: Offset histogram
    if save_dir:
        hist_save_path = os.path.join(save_dir, "elionix_log_alignment_histogram.png")
        plot_alignment_offset_hist(
            dx_nm, dy_nm, save_fig=True, output_path=hist_save_path
        )
    else:
        plot_alignment_offset_hist(
            dx_nm, dy_nm, save_fig=save_fig, output_path=offset_hist_path
        )


if __name__ == "__main__":
    main()
