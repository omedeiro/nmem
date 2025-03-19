import ltspice
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
from nmem.analysis.analysis import CMAP
from nmem.simulation.spice_circuits.functions import (
    import_raw_dir,
    process_read_data,
)
from nmem.simulation.spice_circuits.plotting import (
    plot_current_sweep_ber,
    plot_current_sweep_output,
    plot_current_sweep_persistent,
    plot_transient,
    plot_transient_fill,
)


def example_transient_operation() -> None:
    """
    This function loads the simulation data from the specified RAW file,
    parses it using the ltspice library, and creates a plot showing both
    the drain current (Ix(HR:drain)) and control voltage (V(ichr)) over time.

    Returns:
        None: This function displays a plot but does not return any values.

    File:
        The function uses data from "spice_simulation_raw/single_transient_operation/nmem_cell_write_read_clear.raw"
        This file contains the following sequence of operations:
        1. Write pulse of 50uA
        2. Read operation
        3. Clear operation
        4. Write pulse of -50uA
        5. Read operation
    """
    file_path = "spice_simulation_raw/single_transient_operation/nmem_cell_write_read_clear.raw"
    ltsp = ltspice.Ltspice(file_path)
    ltsp.parse()
    data_dict = process_read_data(ltsp)
    fig, ax = plt.subplots()
    ax = plot_transient(ax, data_dict, signal_name="tran_left_branch_current")
    ax = plot_transient(
        ax, data_dict, signal_name="tran_right_branch_current", color="grey"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    plt.show()


def example_step_read_sweep_write() -> None:
    """
    This function loads the simulation data from the specified RAW files,
    parses it using the ltspice library, and creates a plot showing the
    output voltage (V(ichr)) for a series of read operations with different
    write currents.

    Returns:
        None: This function displays a plot but does not return any values.

    Files:
        The function uses data from the following directory:
        "spice_simulation_raw/read_current_sweep_2/"
        For write currents of 100uA, 140uA, 150uA, 200uA, 250uA
    """
    data_dict = import_raw_dir("spice_simulation_raw/read_current_sweep_2/")
    fig, ax = plt.subplots()
    colors = CMAP(np.linspace(0, 1, len(data_dict)))
    for i, (key, ltsp) in enumerate(data_dict.items()):
        write_current = key[-9:-6]
        processed_data = process_read_data(ltsp)
        ax = plot_current_sweep_output(
            ax, processed_data, color=colors[i], label="Write " + write_current + "uA"
        )
    ax.legend(loc="upper left", ncol=1, prop={"size": 12}, bbox_to_anchor=(1.05, 1))
    plt.show()


def example_transient_branch_currents(case: int = 18) -> None:
    ltsp = ltspice.Ltspice("spice_simulation_raw/nmem_cell_write_1000uA.raw").parse()
    data_dict = process_read_data(ltsp)

    if case > ltsp.case_count:
        raise ValueError(f"Case {case} not found in data")

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    ax: plt.Axes = axs[0]
    plot_transient(
        ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name="tran_left_branch_current",
        color="grey",
    )
    plot_transient_fill(
        ax,
        data_dict,
        cases=[case],
        s1="tran_left_critical_current",
        s2="tran_left_branch_current",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

    ax: plt.Axes = axs[1]
    plot_transient(
        ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name="tran_right_branch_current",
        color="grey",
    )
    plot_transient_fill(
        ax,
        data_dict,
        cases=[case],
        s1="tran_right_critical_current",
        s2="tran_right_branch_current",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.show()


def example_step_enable_write_sweep_write() -> None:
    data_dict = import_raw_dir("spice_simulation_raw/enable_write_sweep/")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    colors = CMAP(np.linspace(0, 1, len(data_dict)))
    for i, (key, ltsp) in enumerate(data_dict.items()):
        processed_data = process_read_data(ltsp)
        write_current = key[-9:-6]
        ax = axs[0]
        ax = plot_current_sweep_ber(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )
        ax.set_ylabel("BER")

        ax = axs[1]
        ax = plot_current_sweep_output(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )
        ax.set_ylabel("Output Voltage (mV)")

        ax = axs[2]
        ax = plot_current_sweep_persistent(
            ax,
            processed_data,
            label=f"Write current {write_current}uA",
            color=colors[i],
        )

    for ax in axs:

        ax.legend(loc="upper left", ncol=1, prop={"size": 12}, bbox_to_anchor=(1.05, 1))
    plt.show()


def example_transient_write_sweep() -> None:
    ltsp = ltspice.Ltspice(
        "spice_simulation_raw/write_current_sweep/nmem_cell_write_200uA.raw"
    ).parse()
    data_dict = process_read_data(ltsp)

    frame_path = os.path.join(
        os.getcwd(), "spice_simulation_raw", "write_current_sweep", "frames"
    )
    os.makedirs(frame_path, exist_ok=True)
    frame_filenames = []

    for case in range(0, ltsp.case_count, 20):
        write_current = data_dict[case]["write_current"]
        write_current = write_current[case]
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax: plt.Axes = axs[0]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_left_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_left_critical_current",
            s2="tran_left_branch_current",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax: plt.Axes = axs[1]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_right_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_right_critical_current",
            s2="tran_right_branch_current",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.text(0.1, 0.8, f"write current {write_current}uA", transform=ax.transAxes)
        frame_filename = f"{frame_path}/frame_{case}.png"
        plt.savefig(frame_filename)
        frame_filenames.append(frame_filename)
        plt.close(fig)

    # Create GIF
    save_gif = True
    if save_gif:
        gif_filename = frame_path + "/write_current_sweep.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=0.2, loop=0) as writer:
            for filename in frame_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved as {gif_filename}")


def example_transient_write_sweep2() -> None:
    ltsp = ltspice.Ltspice(
        "spice_simulation_raw/write_current_sweep_2/write_sweep_0_300uA_1uA.raw"
    ).parse()
    data_dict = process_read_data(ltsp)
    save_gif = False

    frame_path = os.path.join(
        os.getcwd(), "spice_simulation_raw", "write_current_sweep_2", "frames"
    )
    os.makedirs(frame_path, exist_ok=True)
    frame_filenames = []

    for case in range(0, ltsp.case_count, 20):
        write_current = data_dict[case]["write_current"]
        write_current = write_current[case]
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax: plt.Axes = axs[0]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_left_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_left_critical_current",
            s2="tran_left_branch_current",
        )
        ax.set_ylim(-100, 300)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax: plt.Axes = axs[1]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_right_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_right_critical_current",
            s2="tran_right_branch_current",
        )
        ax.set_ylim(-100, 900)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.text(0.1, 0.8, f"write current {write_current}uA", transform=ax.transAxes)
        
        if save_gif:
            frame_filename = f"{frame_path}/frame_{case}.png"
            plt.savefig(frame_filename)
            frame_filenames.append(frame_filename)
            plt.close(fig)
        else:
            plt.show()

    # Create GIF
    if save_gif:
        gif_filename = frame_path + "/write_current_sweep_3.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=0.2, loop=0) as writer:
            for filename in frame_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved as {gif_filename}")

def example_transient_write_sweep3() -> None:
    ltsp = ltspice.Ltspice(
        "spice_simulation_raw/write_current_sweep_3/nmem_cell_write_step_5uA.raw"
    ).parse()
    data_dict = process_read_data(ltsp)
    save_gif = True

    frame_path = os.path.join(
        os.getcwd(), "spice_simulation_raw", "write_current_sweep_3", "frames"
    )
    os.makedirs(frame_path, exist_ok=True)
    frame_filenames = []

    for case in range(0, ltsp.case_count, 10):
        write_current = data_dict[case]["write_current"]
        write_current = write_current[case]
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax: plt.Axes = axs[0]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_left_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_left_critical_current",
            s2="tran_left_branch_current",
        )
        ax.set_ylim(-100, 300)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax: plt.Axes = axs[1]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_right_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_right_critical_current",
            s2="tran_right_branch_current",
        )
        ax.set_ylim(-100, 900)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.text(0.1, 0.8, f"write current {write_current}uA", transform=ax.transAxes)
        
        if save_gif:
            frame_filename = f"{frame_path}/frame_{case}.png"
            plt.savefig(frame_filename)
            frame_filenames.append(frame_filename)
            plt.close(fig)
        else:
            plt.show()

    # Create GIF
    if save_gif:
        gif_filename = frame_path + "/write_current_sweep_3.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=0.2, loop=0) as writer:
            for filename in frame_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved as {gif_filename}")



def example_step_read_current() -> None:
    ltsp = ltspice.Ltspice("spice_simulation_raw/step_read_current/nmem_cell_read_IER_250uA.raw").parse()
    data_dict = process_read_data(ltsp)

    frame_path = os.path.join(
        os.getcwd(), "spice_simulation_raw", "step_read_current", "frames"
    )
    os.makedirs(frame_path, exist_ok=True)
    frame_filenames = []

    for case in range(0, ltsp.case_count, 20):
        enable_read_current = data_dict[case]["enable_read_current"][case]
        read_current = data_dict[case]["read_current"][case]

        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax: plt.Axes = axs[0]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_left_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_left_critical_current",
            s2="tran_left_branch_current",
        )
        ax.set_ylim(-100, 400)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax: plt.Axes = axs[1]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_right_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_right_critical_current",
            s2="tran_right_branch_current",
        )
        ax.set_ylim(-100, 700)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.text(
            0.1,
            0.8,
            f"enable read current {enable_read_current}uA",
            transform=ax.transAxes,
        )
        frame_filename = f"{frame_path}/frame_{case}.png"
        plt.savefig(frame_filename)
        frame_filenames.append(frame_filename)
        plt.close(fig)

    # Create GIF
    save_gif = False
    if save_gif:
        gif_filename = frame_path + "/step_read_current.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=2, loop=0) as writer:
            for filename in frame_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved as {gif_filename}")


if __name__ == "__main__":
    # example_transient_operation()
    # example_step_read_sweep_write()
    # example_step_enable_write_sweep_write()
    # example_transient_branch_currents()
    example_transient_write_sweep3()
    # example_step_read_current()
