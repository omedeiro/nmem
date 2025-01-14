import os

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    build_array,
    filter_first,
    import_directory,
)
from nmem.measurement.functions import calculate_channel_temperature

if __name__ == "__main__":
    data_list = import_directory(os.getcwd())
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))

    fig, ax = plt.subplots()
    for data in data_list:
        x, y, ztotal = build_array(data, "bit_error_rate")
        ax.plot(
            y,
            ztotal,
            label=f"write current = {filter_first(data.get('write_current'))*1e6}",
            color=colors[data_list.index(data)],
        )
        enable_read_current = data.get("enable_read_current") * 1e6
        cell = data.get("cell")[0]
        max_enable_read_current = (
            filter_first(data["CELLS"][cell][0][0]["x_intercept"])
        )

        enable_read_temp = calculate_channel_temperature(
            1.3, 12.3, enable_read_current, max_enable_read_current
        )

    ax.legend()
    ax.set_ylim([0, 1])
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
