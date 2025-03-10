import ltspice
import numpy as np
import matplotlib.pyplot as plt

CMAP = plt.get_cmap("coolwarm")


def plot_data(
    ax: plt.Axes,
    ltspice_data: ltspice.Ltspice,
    signal_name: str,
    case: int = 0,
    **kwargs,
) -> plt.Axes:
    time = ltspice_data.get_time(case=case)
    signal = ltspice_data.get_data(signal_name, case=case) * 1e6
    ax.plot(time, signal, label=signal_name, **kwargs)
    return ax


def plot_write_read_clear():
    file_path = "spice_simulation_raw/nmem_cell_write_read_clear.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file

    fig, ax = plt.subplots()
    ax = plot_data(ax, l, "Ix(HR:drain)")
    ax = plot_data(ax, l, "V(ichr)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    plt.show()


def get_persistent_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal_l = ltspice_data.get_data("Ix(HL:drain)", case=case)
    signal_r = ltspice_data.get_data("Ix(HR:drain)", case=case)
    return np.array([np.abs(signal_r[-1] - signal_l[-1]) / 2 * 1e6])


def get_write_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("I(R2)", case=case)
    return np.max(signal) * 1e6


def get_irhl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhl)", case=case)[0] * 1e6])


def get_irhr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhr)", case=case)[0] * 1e6])


def get_ichl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichl)", case=case)[0] * 1e6])


def get_ichr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichr)", case=case)[0] * 1e6])


def plot_nmem_cell(file_path: str):
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file
    colors = [CMAP(i) for i in np.linspace(0, 1, l.case_count)]
    fig, ax = plt.subplots()
    persistent_currents = np.zeros(l.case_count)
    write_currents = np.zeros(l.case_count)
    max_outputs = np.zeros(l.case_count)
    for i in range(0, l.case_count):
        ax = plot_data(ax, l, "Ix(HL:drain)", case=i, color=colors[i])
        ax = plot_data(ax, l, "Ix(HR:drain)", case=i, color=colors[i], linestyle="--")
        ax = plot_data(ax, l, "V(ichl)", case=i, color="k", linestyle="-.")
        ax = plot_data(ax, l, "V(ichr)", case=i, color="k", linestyle=":")

        persistent_currents[i] = get_persistent_current(l, case=i)
        write_currents[i] = get_write_current(l, case=i)
        max_outputs[i] = get_max_output(l, case=i)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    fig.savefig("nmem_cell_write.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(write_currents, persistent_currents, "-o")

    ax2 = ax.twinx()
    ax2.plot(write_currents, max_outputs * 1e3, "-o", color="r", label="Voltage Output")
    ax2.set_ylabel("Peak Voltage Output (mV)")

    ax.plot([0, ax.get_ylim()[1] * 2], [0, ax.get_ylim()[1]], "--", label="y=x/2")

    ax.legend()
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")

    fig.savefig(f"nmem_cell_write_slope.png")
    plt.show()


def get_max_output(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("V(out)", case=case)
    return np.max(signal)


def plot_all_write_sweeps():
    plot_nmem_cell("spice_simulation_raw/nmem_cell_write_200uA.raw")
    plot_nmem_cell("spice_simulation_raw/nmem_cell_write_1000uA.raw")
    plot_nmem_cell("spice_simulation_raw/nmem_cell_write_step_5uA.raw")
    plot_nmem_cell("spice_simulation_raw/nmem_cell_write_step_10uA.raw")


def plot_nmem_cell_read():
    file_name = "nmem_cell_read.raw"
    l = ltspice.Ltspice(file_name)
    l.parse()  # Parse the raw file
    fig, ax = plt.subplots()
    # ax = plot_data(ax, l, "V(ichr)")
    time = l.get_time()
    irhl = get_irhl(l)
    irhr = get_irhr(l)
    ichl = get_ichl(l)
    ichr = get_ichr(l)
    # ax.plot(time, irhl, label="IRHL")
    # ax = plot_data(ax, l, "Ix(HL:drain)")
    # ax = plot_data(ax, l, "Ix(HR:drain)")


READ_ONE_START = 2e-7
READ_ONE_END = 2.5e-7
READ_ZERO_START = 6e-7
READ_ZERO_END = 6.5e-7

if __name__ == "__main__":
    file_name = "nmem_cell_read.raw"
    l = ltspice.Ltspice(file_name)
    l.parse()  # Parse the raw file
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, l.case_count)]
    fig, axs = plt.subplots(l.case_count, 1, figsize=(8, 4 * l.case_count))
    read_output = np.zeros((l.case_count, 2))
    read_current = np.zeros((l.case_count, 1))
    for i in range(0, l.case_count):
        ax = axs[i]
        time = l.get_time(i)
        channel_current = l.get_data("I(R2)", i) * 1e6
        output_voltage = l.get_data("V(out)", i)

        read_one_time = (time > READ_ONE_START) & (time < READ_ONE_END)
        read_one_voltage = np.max(
            output_voltage[read_one_time]
        )
        read_zero_time = (time > READ_ZERO_START) & (time < READ_ZERO_END)
        read_zero_voltage = np.max(
            output_voltage[read_zero_time]
        )
        read_output[i, 0] = read_zero_voltage
        read_output[i, 1] = read_one_voltage
        read_current[i] = np.max(channel_current[read_one_time])
        ax.plot(time, channel_current, label=f"Channel Current {i}", color=colors[i])
        ax2 = ax.twinx()
        ax2.plot(
            time, output_voltage, label=f"Output Voltage {i}", color="k", linestyle="--"
        )
        ax2.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig, ax = plt.subplots()
    ax.plot(read_current, read_output[:, 0], "-o", label="Read 0")
    ax.plot(read_current, read_output[:, 1], "-o", label="Read 1")
    ax.legend()
    ax.set_ylabel("Output Voltage (V)")
    ax.set_xlabel("Case")
