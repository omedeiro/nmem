import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.constants import WAFER_RADIUS
from nmem.analysis.core_analysis import interpolate_map
from nmem.analysis.data_import import load_and_clean_thickness
from nmem.analysis.matrix_plots import plot_wafer_maps
from nmem.analysis.utils import generate_boundary_points

GRID_RES = 400j
NUM_BOUNDARY_POINTS = 100
CSV_BEFORE = "../data/wafer_thickness_mapping/MapResult_623.csv"
CSV_AFTER = "../data/wafer_thickness_mapping/MapResult_624.csv"


def main(
    csv_before=CSV_BEFORE,
    csv_after=CSV_AFTER,
    wafer_radius=WAFER_RADIUS,
    grid_res=GRID_RES,
    num_boundary_points=NUM_BOUNDARY_POINTS,
    save_fig=False,
    output_path="wafer_maps_before_after_delta.pdf",
):
    before_map = load_and_clean_thickness(csv_before)
    after_map = load_and_clean_thickness(csv_after)
    delta_map = after_map - before_map

    grid_x, grid_y = np.mgrid[
        -wafer_radius:wafer_radius:grid_res, -wafer_radius:wafer_radius:grid_res
    ]

    before_boundary = generate_boundary_points(
        wafer_radius, num_boundary_points, np.nanmean(before_map.values)
    )
    after_boundary = generate_boundary_points(
        wafer_radius, num_boundary_points, np.nanmean(after_map.values)
    )
    delta_boundary = generate_boundary_points(
        wafer_radius, num_boundary_points, np.nanmean(delta_map.values)
    )

    before_interp = interpolate_map(
        before_map, wafer_radius, grid_x, grid_y, before_boundary
    )
    after_interp = interpolate_map(
        after_map, wafer_radius, grid_x, grid_y, after_boundary
    )
    delta_interp = interpolate_map(
        delta_map, wafer_radius, grid_x, grid_y, delta_boundary
    )

    fig = plot_wafer_maps(
        [before_interp, after_interp, delta_interp],
        ["Before Deposition", "After Deposition", "Deposited Thickness"],
        [plt.cm.Blues, plt.cm.Greens, plt.cm.inferno],
        grid_x,
        grid_y,
        wafer_radius,
    )
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
