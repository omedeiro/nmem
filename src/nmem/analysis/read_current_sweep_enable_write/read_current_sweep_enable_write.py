import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep,
    plot_state_current_markers,
    get_state_currents_measured
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data_list = import_directory("data")
    fig, ax = plt.subplots()
    for data_dict in [data_list[-4]]:
        plot_read_sweep(ax, data_dict, "bit_error_rate", "enable_write_current")
        plot_state_current_markers(ax, data_dict)
        state_currents = get_state_currents_measured(data_dict)
        print(state_currents)
        print(f"")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Write Current",
    )
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    plt.show()
