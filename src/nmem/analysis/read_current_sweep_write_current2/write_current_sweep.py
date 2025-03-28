import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
    get_channel_temperature
)


def plot_read_temp_sweep_C3(save=True):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    dict_list = [
        import_directory("write_current_sweep_C3_1"),
        import_directory("write_current_sweep_C3_3"),
        import_directory("write_current_sweep_C3_4"),
        import_directory("write_current_sweep_C3"),
    ]
    for i, data_dict in enumerate(dict_list):
        enable_temperature = get_channel_temperature(data_dict[0], "read")
        plot_read_sweep_array(
            axs[i // 2, i % 2],
            data_dict,
            "bit_error_rate",
            "write_current",
            marker=".",
        )
        axs[i // 2, i % 2].set_xlabel("Read Current [$\mu$A]")
        axs[i // 2, i % 2].set_ylabel("Bit Error Rate")
        axs[i // 2, i % 2].set_title(f"Read Temperature: {enable_temperature:0.2f} $\mu$A")
   
    axs[0, 1].legend(
        frameon=False,
        bbox_to_anchor=(1.1, 1),
        loc="upper left",
        title="Write Current [$\mu$A]",
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    if save:
        plt.savefig("read_current_sweep_write_current_C3.pdf", bbox_inches="tight")

def plot_read_sweep_import(data_dict: dict[str, list[float]]):
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_current")
    cell = data_dict[0]["cell"][0]

    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    ax.set_title(f"Cell {cell}")
    return fig, ax

if __name__ == "__main__":
    # plot_read_sweep_import(import_directory("write_current_sweep_B2_0"))
    # plot_read_sweep_import(import_directory("write_current_sweep_B2_1"))
    # plot_read_sweep_import(import_directory("write_current_sweep_B2_2"))

    # plot_read_sweep_import(import_directory("write_current_sweep_A2"))
    # plot_read_sweep_import(import_directory("write_current_sweep_C2"))


    plot_read_temp_sweep_C3()

    