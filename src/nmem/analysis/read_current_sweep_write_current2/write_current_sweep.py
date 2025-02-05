import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)


def plot_read_temp_sweep_C3(save=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    plot_read_sweep_array(
        axs[0, 0],
        import_directory("write_current_sweep_C3_2"),
        "bit_error_rate",
        "write_current",
    )
    axs[0, 0].set_xlabel("Read Current [$\mu$A]")
    axs[0, 0].set_ylabel("Bit Error Rate")
    plot_read_sweep_array(
        axs[0, 1],
        import_directory("write_current_sweep_C3_3"),
        "bit_error_rate",
        "write_current",
    )
    axs[0, 1].set_xlabel("Read Current [$\mu$A]")
    axs[0, 1].set_ylabel("Bit Error Rate")
    plot_read_sweep_array(
        axs[1, 0],
        import_directory("write_current_sweep_C3_4"),
        "bit_error_rate",
        "write_current",
    )
    axs[1, 0].set_xlabel("Read Current [$\mu$A]")
    axs[1, 0].set_ylabel("Bit Error Rate")
    plot_read_sweep_array(
        axs[1, 1],
        import_directory("write_current_sweep_C3"),
        "bit_error_rate",
        "write_current",
    )
    axs[1, 1].set_xlabel("Read Current [$\mu$A]")
    axs[1, 1].set_ylabel("Bit Error Rate")
    axs[0, 1].legend(
        frameon=False,
        bbox_to_anchor=(1.1, 1),
        loc="upper left",
        title="Write Current [$\mu$A]",
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    if save:
        plt.savefig(
            "read_current_sweep_write_current_C3.png", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax,
        import_directory("write_current_sweep_B2_0"),
        "bit_error_rate",
        "write_current",
    )
    plot_read_sweep_array(
        ax,
        import_directory("write_current_sweep_B2_1"),
        "bit_error_rate",
        "write_current",
    )
    plot_read_sweep_array(
        ax,
        import_directory("write_current_sweep_B2_2"),
        "bit_error_rate",
        "write_current",
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax,
        import_directory("write_current_sweep_A2"),
        "bit_error_rate",
        "write_current",
    )
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    plt.show()

    fig, ax = plt.subplots()
    plot_read_sweep_array(
        ax,
        import_directory("write_current_sweep_C2"),
        "bit_error_rate",
        "write_current",
    )
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    plt.show()

    plot_read_temp_sweep_C3()
