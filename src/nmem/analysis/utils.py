import collections.abc
from typing import Any, Literal, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from nmem.analysis.bit_error import (
    get_total_switches_norm,
)


def filter_plateau(
    xfit: np.ndarray, yfit: np.ndarray, plateau_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    xfit = np.where(yfit < plateau_height, xfit, np.nan)
    yfit = np.where(yfit < plateau_height, yfit, np.nan)

    # Remove nans
    xfit = xfit[~np.isnan(xfit)]
    yfit = yfit[~np.isnan(yfit)]

    return xfit, yfit


def filter_first(value) -> Any:
    if isinstance(value, collections.abc.Iterable) and not isinstance(
        value, (str, bytes)
    ):
        return np.asarray(value).flatten()[0]
    return value


def filter_nan(x, y):
    mask = np.isnan(y)
    x = x[~mask]
    y = y[~mask]
    return x, y


def convert_cell_to_coordinates(cell: str) -> tuple:
    """Converts a cell name like 'A1' to coordinates (x, y)."""
    column_letter = cell[0]
    row_number = int(cell[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def get_current_cell(data_dict: dict) -> str:
    cell = filter_first(data_dict.get("cell"))
    if cell is None:
        cell = filter_first(data_dict.get("sample_name"))[-2:]
    return cell


def build_array(
    data_dict: dict, parameter_z: Literal["total_switches_norm"]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data_dict.get("total_switches_norm") is None:
        data_dict["total_switches_norm"] = get_total_switches_norm(data_dict)
    x: np.ndarray = data_dict.get("x")[0][:, 0] * 1e6
    y: np.ndarray = data_dict.get("y")[0][:, 0] * 1e6
    z: np.ndarray = data_dict.get(parameter_z)

    xlength: int = filter_first(data_dict.get("sweep_x_len", len(x)))
    ylength: int = filter_first(data_dict.get("sweep_y_len", len(y)))

    # X, Y reversed in reshape
    zarray = z.reshape((ylength, xlength), order="F")
    return x, y, zarray


def build_resistance_map(df, grid_size=56):
    """Build a resistance map from a DataFrame with x_abs, y_abs, Rmean."""
    Rmeas = np.full((grid_size, grid_size), np.nan)
    for _, row in df.iterrows():
        x, y = row["x_abs"], row["y_abs"]
        if 0 <= x < grid_size and 0 <= y < grid_size:
            val = row["Rmean"]
            Rmeas[int(x), int(y)] = 0 if val < 0 else val
    Rmeas[Rmeas == 0] = np.nanmax(Rmeas)
    return Rmeas


def normalize_row_by_squares(Rmeas, row_letter, length_um=300):
    """Normalize resistance values in a row by number of squares."""
    row_map = {k: i for i, k in enumerate("ABCDEFG")}
    y_start = row_map[row_letter.upper()] * 8
    length_nm = length_um * 1e3
    Rmeas = Rmeas.copy()

    for i in range(8):
        width_nm = 50 * (i + 1)
        num_squares = length_nm / width_nm
        Rmeas[:, y_start + i] /= num_squares

    return Rmeas


def create_rmeas_matrix(df, x_col, y_col, value_col, shape):
    """Create a 2D resistance matrix from DataFrame columns."""
    Rmeas = np.full(shape, np.nan)
    for _, row in df.iterrows():
        x, y = int(row[x_col]), int(row[y_col])
        val = row[value_col]
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            Rmeas[y, x] = np.nan if val < 0 else val
    return Rmeas


def summarize_die_yield(df, wafer_rows, min_kohm=1, max_kohm=50000):
    df = df.copy()
    df["Rmean_k"] = df["Rmean"] / 1e3

    summary_records = []

    for row_num in wafer_rows:
        row_df = df[df["die"].str.endswith(row_num)].copy()
        row_df["is_outlier"] = (
            row_df["Rmean_k"].isna()
            | (row_df["Rmean_k"] < min_kohm)
            | (row_df["Rmean_k"] > max_kohm)
        )

        grouped = row_df.groupby("die")
        die_outlier_counts = grouped["is_outlier"].sum().astype(int)
        die_total_counts = grouped["is_outlier"].count()

        for die in die_outlier_counts.index:
            n_bad = die_outlier_counts[die]
            n_total = die_total_counts[die]
            yield_pct = 100 * (1 - n_bad / n_total) if n_total > 0 else np.nan
            summary_records.append(
                {
                    "row": row_num,
                    "die": die,
                    "total_devices": n_total,
                    "outliers": n_bad,
                    "yield_percent": yield_pct,
                }
            )

    summary_df = pd.DataFrame(summary_records)

    # Add row-level statistics
    row_stats = (
        summary_df.groupby("row")["yield_percent"]
        .agg(
            row_mean_yield="mean",
            row_std_yield="std",
            row_min_yield="min",
            row_max_yield="max",
        )
        .reset_index()
    )

    return summary_df, row_stats


def generate_boundary_points(radius, n_points, fill_value):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    bx = radius * np.cos(angles)
    by = radius * np.sin(angles)
    bz = np.full_like(bx, fill_value)
    return bx, by, bz


def center_crop_zoom(img, zoom_factor=2):
    """Rotate by 90 degrees, then crop the center and zoom in by the given factor."""
    rotated = img.rotate(-90)  # counterclockwise
    w, h = rotated.size
    crop_w, crop_h = int(w / zoom_factor), int(h / zoom_factor)

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    cropped = rotated.crop((left, top, right, bottom))
    zoomed = cropped.resize((w, h), Image.LANCZOS)
    return zoomed


def get_cell_labels(rows=None, cols=None):
    """
    Generate cell labels like A1, B1, ..., D4 by default.
    Optionally specify rows and cols for custom grids.
    """
    if rows is None:
        rows = ["A", "B", "C", "D"]
    if cols is None:
        cols = range(1, 5)
    return [f"{r}{c}" for r in rows for c in cols]
