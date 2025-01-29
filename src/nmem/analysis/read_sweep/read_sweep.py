import matplotlib.pyplot as plt
from nmem.analysis.analysis import (
    import_directory,
    plot_voltage_trace_stack,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 10


if __name__ == "__main__":
    dict_list = import_directory("data")

    for data_dict in dict_list:
        fig, axs = plt.subplots(3, 1)
        plot_voltage_trace_stack(axs, data_dict)
        plt.show()
