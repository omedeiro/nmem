

from nmem.analysis.bar_plots import (
    plot_alignment_offset_hist,
)
from nmem.analysis.core_analysis import analyze_alignment_stats
from nmem.analysis.data_import import import_elionix_log
from nmem.analysis.bar_plots import (
    plot_alignment_stats,
)


def main(
    log_path="New schedule.log",
    save=False,
    save_fig=False,
    analysis_path="alignment_analysis.pdf",
    offset_hist_path="alignment_offsets_histogram.pdf",
):
    """
    Main function to import, analyze, and plot Elionix alignment log data.
    """

    df_z, df_rot_valid, dx_nm, dy_nm, delta_table = import_elionix_log(log_path)
    z_mean, z_std, r_mean, r_std = analyze_alignment_stats(
        df_z, df_rot_valid, dx_nm, dy_nm
    )
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
    print(delta_table)
    plot_alignment_offset_hist(
        dx_nm, dy_nm, save_fig=save_fig, output_path=offset_hist_path
    )


if __name__ == "__main__":
    main()
