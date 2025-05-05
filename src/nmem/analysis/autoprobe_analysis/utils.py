import numpy as np


def create_rmeas_matrix(df, x_col, y_col, value_col, shape):
    """Create a 2D resistance matrix from DataFrame columns."""
    Rmeas = np.full(shape, np.nan)
    for _, row in df.iterrows():
        x, y = int(row[x_col]), int(row[y_col])
        val = row[value_col]
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            Rmeas[y, x] = np.nan if val < 0 else val
    return Rmeas


def get_log_norm_limits(R):
    """Safely get vmin and vmax for LogNorm."""
    values = R[~np.isnan(R) & (R > 0)]
    if values.size == 0:
        return None, None
    return np.nanmin(values), np.nanmax(values)


def annotate_matrix(ax, R, fmt="{:.2g}", color="white"):
    """Add text annotations to matrix cells."""
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            val = R[y, x]
            if not np.isnan(val):
                ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=6, color=color)
