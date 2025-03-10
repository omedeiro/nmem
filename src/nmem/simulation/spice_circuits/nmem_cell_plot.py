import ltspice
import numpy as np
import matplotlib.pyplot as plt
CMAP = plt.get_cmap("coolwarm")

def plot_data(
    ax: plt.Axes, ltspice_data: ltspice.Ltspice, signal_name: str, case: int = 0, **kwargs
) -> plt.Axes:
    time = ltspice_data.get_time(case=case)
    signal = ltspice_data.get_data(signal_name, case=case)
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
    print(f"signal_l: {signal_l[-1]*1e6}, signal_r: {signal_r[-1]*1e6}")
    return np.array([signal_r[-1]*1e6])

def get_write_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("I(R2)", case=case)
    return np.max(signal)*1e6

def plot_nmem_cell():
    file_path = "nmem_cell_write.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file
    colors = [CMAP(i) for i in np.linspace(0, 1, l.case_count)]
    fig, ax = plt.subplots()
    persistent_currents = np.zeros(l.case_count)
    write_currents = np.zeros(l.case_count)
    for i in range(0, l.case_count):
        ax = plot_data(ax, l, "Ix(HL:drain)", case=i, color=colors[i])
        ax = plot_data(ax, l, "Ix(HR:drain)", case=i, color=colors[i], linestyle="--")
        persistent_currents[i] = get_persistent_current(l, case=i)
        write_currents[i] = get_write_current(l, case=i)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")

    fig, ax = plt.subplots()
    ax.plot(write_currents, persistent_currents, "-o")

    ax.plot([0, 300], [0, 150], "r--", label="y = x/2")
    ax.legend()
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")

if __name__ == "__main__":
    plot_nmem_cell()
