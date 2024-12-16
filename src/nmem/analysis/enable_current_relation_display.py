import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import import_directory, plot_enable_current_relation

plt.rcParams["figure.figsize"] = [5.5, 5.5]


def display_all(file_path: str):
    dict_list = import_directory(file_path)
    for dict in dict_list:
        cell = dict.get("cell")
        if isinstance(cell, np.ndarray):
            cell = cell.flat[0]
        if cell is None:
            cell = "_"

        fig, ax = plt.subplots()
        plot_enable_current_relation(ax, dict)
        ax.set_xlabel("Enable Current [$\mu$A]")
        ax.set_ylabel("Critical Current [$\mu$A]")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Cell {cell}")


if __name__ == "__main__":
    # display_all("data")
    display_all("data2")
