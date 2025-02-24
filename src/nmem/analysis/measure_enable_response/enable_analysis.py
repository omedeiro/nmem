import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_column,
    plot_full_grid,
    plot_grid,
    plot_row,
)

if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)
    plot_full_grid(axs, dict_list)
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_column(axs, dict_list)
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    plot_row(axs, dict_list)
    plt.show()

    fig, axs = plt.subplots(4, 4, figsize=(20, 20), sharex=True, sharey=True)
    plot_grid(axs, dict_list)
    plt.show()
