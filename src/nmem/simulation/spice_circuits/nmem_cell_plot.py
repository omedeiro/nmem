import ltspice
import numpy as np
import matplotlib.pyplot as plt
import os

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


def process_read_data(l: ltspice.Ltspice):
    read_output = np.zeros((l.case_count, 2))
    read_current = np.zeros((l.case_count, 1))
    for i in range(0, l.case_count):
        time = l.get_time(i)
        enable_current = l.get_data("I(R1)", i) * 1e6
        channel_current = l.get_data("I(R2)", i) * 1e6
        output_voltage = l.get_data("V(out)", i)

        read_one_time = (time > READ_ONE_START) & (time < READ_ONE_END)
        read_one_voltage = np.max(output_voltage[read_one_time])
        read_zero_time = (time > READ_ZERO_START) & (time < READ_ZERO_END)
        read_zero_voltage = np.max(output_voltage[read_zero_time])
        read_output[i, 0] = read_zero_voltage
        read_output[i, 1] = read_one_voltage
        read_current[i] = np.max(channel_current[read_one_time])
        enable_read_current = np.max(enable_current[read_one_time])

    return {
        "read_current": read_current,
        "read_output": read_output,
        "enable_read_current": enable_read_current,
    }


def import_read_data():
    read_data = np.genfromtxt(
        "spice_simulation_raw/read_data_processed.csv", delimiter=","
    )
    read_current = read_data[:, 0]
    read_output = read_data[:, 1:]
    return read_current, read_output


def save_data_file(l: ltspice.Ltspice):
    read_outputs = process_read_data(l)
    enable_read_current = read_outputs["enable_read_current"]
    read_current = read_outputs["read_current"]
    read_output = read_outputs["read_output"]
    np.savetxt(
        f"spice_simulation_raw/read_data_processed_{enable_read_current:.0f}uA.csv",
        np.hstack((read_current, read_output)),
        delimiter=",",
        header="read_current, read_output_0, read_output_1",
    )


def import_csv_dir(file_path: str) -> dict:
    # get only csv from directory
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    files.sort()
    data_dict = {}
    for file in files:
        enable_read_current = float(file[-9:-6])
        data = np.genfromtxt(file_path + file, delimiter=",")
        data_dict[enable_read_current] = data

    return data_dict


def plot_read_current_output(
    ax: plt.Axes,
    read_current: np.ndarray,
    read_output: np.ndarray,
    enable_read_current: float,
) -> plt.Axes:
    ax.plot(read_current, read_output[:, 0] * 1e3, "-o", label="Read 0")
    ax.plot(read_current, read_output[:, 1] * 1e3, "-o", label="Read 1")
    ax.text(
        0.1, 0.5, f"Enable Current: {enable_read_current:.0f}uA", transform=ax.transAxes
    )
    ax.legend()
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel("Read Current (uA)")
    return ax

def plot_read_data_dict(data_dict: dict):
    fig, ax = plt.subplots()
    for key, data in data_dict.items():
        read_current = data[:, 0]
        read_output = data[:, 1:]
        ax.plot(read_current, read_output[:, 0] * 1e3, "-o", label=f"Read 0, {key:.0f}uA")
        ax.plot(read_current, read_output[:, 1] * 1e3, "-o", label=f"Read 1, {key:.0f}uA")

    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel("Read Current (uA)")
    plt.show()

def process_data_dict(data_dict: dict):
    read_output_0 = data_dict.get("read_output_0")
    read_output_1 = data_dict.get("read_output_1")
    read_current = data_dict.get("read_current")
    voltage_threshold = 2.0 #mV 
    zero_switch = np.argwhere(read_output_0 > voltage_threshold)[0]
    one_switch = np.argwhere(read_output_1 > voltage_threshold)[0]
    zero_current = read_current[zero_switch]
    one_current = read_current[one_switch]
    return zero_current, one_current
    
READ_ONE_START = 2e-7
READ_ONE_END = 2.5e-7
READ_ZERO_START = 6e-7
READ_ZERO_END = 6.5e-7
VOUT_YMAX = 40

if __name__ == "__main__":
    file_path = "nmem_cell_read.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file
    # save_data_file(l)
    read_outputs = process_read_data(l)
    enable_read_current = read_outputs["enable_read_current"]
    read_current = read_outputs["read_current"]
    read_output = read_outputs["read_output"]

    fig, ax = plt.subplots()
    ax = plot_read_current_output(ax, read_current, read_output, enable_read_current)


    data_dict = import_csv_dir("spice_simulation_raw/")

    plot_read_data_dict(data_dict)