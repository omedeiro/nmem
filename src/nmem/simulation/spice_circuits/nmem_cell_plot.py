import ltspice
import numpy as np
import matplotlib.pyplot as plt
CMAP = plt.get_cmap("coolwarm")

def plot_data(
    ax: plt.Axes, ltspice_data: ltspice.Ltspice, signal_name: str, case: int = 0, **kwargs
) -> plt.Axes:
    time = ltspice_data.get_time(case=case)
    signal = ltspice_data.get_data(signal_name, case=case)*1e6
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
    return np.array([signal_r[-1]*1e6])

def get_write_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("I(R2)", case=case)
    return np.max(signal)*1e6

def get_irhl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhl)", case=case)[0]*1e6])

def get_irhr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhr)", case=case)[0]*1e6])

def get_ichl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichl)", case=case)[0]*1e6])

def get_ichr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichr)", case=case)[0]*1e6])


def plot_nmem_cell():
    file_path = "spice_simulation_raw/nmem_cell_write_step_5uA.raw"
    l = ltspice.Ltspice(file_path)
    l.parse()  # Parse the raw file
    colors = [CMAP(i) for i in np.linspace(0, 1, l.case_count)]
    fig, ax = plt.subplots()
    persistent_currents = np.zeros(l.case_count)
    write_currents = np.zeros(l.case_count)
    for i in range(0, l.case_count):
        ax = plot_data(ax, l, "Ix(HL:drain)", case=i, color=colors[i])
        ax = plot_data(ax, l, "Ix(HR:drain)", case=i, color=colors[i], linestyle="--")
        ax = plot_data(ax, l, "V(ichl)", case=i, color="k", linestyle="-.")
        ax = plot_data(ax, l, "V(ichr)", case=i, color='k', linestyle=":")

        persistent_currents[i] = get_persistent_current(l, case=i)
        write_currents[i] = get_write_current(l, case=i)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    fig.savefig("nmem_cell_write.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(write_currents, persistent_currents, "-o")
    
    
    ax.axhline(get_irhl(l), color="k", linestyle="--", label="IRHL")
    ax.axhline(get_irhr(l), color="k", linestyle="-.", label="IRHR")
    ax.axhline(get_ichl(l), color="k", linestyle=":", label="ICHL")
    ax.axvline(get_ichr(l), color="k", linestyle="-", label="ICHR")
    ax.plot([0, ax.get_ylim()[1]*2], [0, ax.get_ylim()[1]], "--", label="y=x/2")

    ax.legend()
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")

    fig.savefig(f"nmem_cell_write_slope.png")
    plt.show()

    
if __name__ == "__main__":
    plot_nmem_cell()
