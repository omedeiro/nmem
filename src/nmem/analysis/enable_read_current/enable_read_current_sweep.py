import matplotlib.pyplot as plt
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.markeredgewidth"] = 0.5

plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1


if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, enable_read_290_list, "bit_error_rate", "enable_read_current"
    )
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, enable_read_300_list, "bit_error_rate", "enable_read_current"
    )
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, enable_read_310_list, "bit_error_rate", "enable_read_current"
    )
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax, enable_read_310_C4_list, "bit_error_rate", "enable_read_current"
    )
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_inverse, "bit_error_rate", "enable_write_current")
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data, "bit_error_rate", "enable_read_current")
    plt.show()
