import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_autoprobe_data(filepath, grid_size=56):
    """Load autoprobe data from a parsed .mat file and return as a DataFrame with bounds-checked coordinates."""
    mat = loadmat(filepath, struct_as_record=False, squeeze_me=True)
    die_name = mat["die_name"].flatten()
    device_name = mat["device_name"].flatten()
    data = mat["data"]

    Rmean = data.Rmean.flatten()
    Rmse = data.Rmse.flatten()

    records = []

    for die, dev, rmean, rmse in zip(die_name, device_name, Rmean, Rmse):
        try:
            die_str = str(die)
            dev_str = str(dev)

            # Parse and flip die/device coordinates
            x_die = ord(die_str[0].upper()) - ord("A")  # 'A' → 0
            y_die = 6 - (int(die_str[1]) - 1)           # '1' → 5, '7' → 0

            x_dev = ord(dev_str[0].upper()) - ord("A")  # 'A' → 0
            y_dev = 7 - (int(dev_str[1]) - 1)           # '1' → 7 → 0

            x_abs = x_die * 8 + x_dev
            y_abs = y_die * 8 + y_dev

            # Bounds and value checks
            if not (0 <= x_abs < grid_size and 0 <= y_abs < grid_size):
                raise ValueError(f"Out of bounds: ({x_abs}, {y_abs})")

            if not np.isfinite(rmean) or rmean < 0:
                continue  # skip bad resistance values

            if y_die == 5:
                squares = 50 * (x_dev + 1)
            else:
                squares = None
            records.append({
                "id": f"{die_str}_{dev_str}",
                "die": die_str,
                "device": dev_str,
                "x_die": x_die,
                "y_die": y_die,
                "x_dev": x_dev,
                "y_dev": y_dev,
                "x_abs": x_abs,
                "y_abs": y_abs,
                "Rmean": rmean,
                "Rmse": rmse,
                "squares": squares,
            })

        except Exception as e:
            print(f"Skipping malformed entry: die={die}, dev={dev}, error={e}")

    return pd.DataFrame(records)

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
