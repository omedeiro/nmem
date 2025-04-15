import numpy as np
from matplotlib.colors import LogNorm


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


def annotate_matrix(ax, R, fmt="{:.0f}", color="white"):
    """Add text annotations to matrix cells."""
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            val = R[y, x]
            if not np.isnan(val):
                ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=6, color=color)


def build_device_lookup(die_name, device_name, Rmean, grid_size=56):
    """
    Build a nested dictionary: main_dict[die][device] = {x, y, resistance, type}
    Adds 'type' field: 'wire' for row B, 'squid' for row D
    """
    main_dict = {}

    for i in range(len(die_name)):
        die = die_name[i].upper()
        dev = device_name[i].upper()
        resistance = Rmean[i]
        print(f"Processing: die={die}, dev={dev}, resistance={resistance}")
        try:
            # Die position
            die_row_letter = die[0]
            die_col_number = int(die[1])

            dev_row = ord(dev[0]) - ord("A")      # A–H → 0–7
            dev_col = int(dev[1]) - 1             # 1–8 → 0–7

            x = (die_col_number - 1) * 8 + dev_col
            y = (ord(die_row_letter) - ord("A")) * 8 + dev_row

            if not (0 <= x < grid_size and 0 <= y < grid_size):
                continue
            
            dev_width = None
            # Determine device type based on die row
            if die_row_letter == "B":
                dev_type = "wire"
                dev_width = 50 * (dev_col + 1)
            elif die_row_letter == "D":
                dev_type = "squid"
            else:
                dev_type = "unknown"

            if die not in main_dict:
                main_dict[die] = {}

            main_dict[die][dev] = {
                "x": x,
                "y": y,
                "resistance": resistance if resistance >= 0 else None,
                "type": dev_type,
            }
        
            if dev_width is not None:
                main_dict[die][dev]["width"] = dev_width


        except Exception as e:
            print(f"Skipping invalid entry: die={die}, dev={dev}, error={e}")

    return main_dict
