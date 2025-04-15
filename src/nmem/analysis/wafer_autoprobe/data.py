import numpy as np
from scipy.io import loadmat


def load_autoprobe_data(filepath):
    """Load autoprobe data from a parsed .mat file."""
    mat = loadmat(filepath, squeeze_me=True)
    die_name = mat["die_name"]
    device_name = mat["device_name"]
    data = mat["data"]

    def to_array(raw):
        return np.array([r.item() if np.size(r) == 1 else np.nan for r in raw.flatten()[0]])

    Rmean = to_array(data["Rmean"])
    Rmse = to_array(data["Rmse"])

    return die_name, device_name, Rmean, Rmse


def build_resistance_map(die_name, device_name, Rmean, grid_size=56):
    """Construct the 2D resistance map."""
    Rmeas = np.full((grid_size, grid_size), np.nan)
    for i in range(len(die_name)):
        d, dev = die_name[i], device_name[i]
        x = (ord(d[0]) - 65) * 8 + (ord(dev[0]) - 65)
        y = (8 - int(d[1])) * 8 + (8 - int(dev[1]))
        val = Rmean[i]
        Rmeas[x, y] = 0 if val < 0 else val
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
