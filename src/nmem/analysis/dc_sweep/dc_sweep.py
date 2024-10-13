import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path= r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 2.36]
plt.rcParams["font.size"] = 5.5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams['font.family'] = "Inter"
# plt.rcParams['font.sans-serif'] = 

def import_data(file_path: str):
    data = sio.loadmat(file_path)
    return data

def import_directory(file_path: str):
    data_list = []
    files = get_file_names(file_path)
    for file in files:
        data = import_data(os.path.join(file_path, file))
        data_list.append(data)
    return data_list

def get_file_names(file_path: str):
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith(".mat")]
    return files

def plot_iv_curve(data, ax, **kwargs):
    time = data["trace"][0,:]
    voltage = data["trace"][1,:]

    M = int(np.round(len(voltage), -2))
    currentQuart = np.linspace(0, data['vpp'] / 2 / 10e3, M // 4)
    current = np.concatenate([-currentQuart, np.flip(-currentQuart), currentQuart, np.flip(currentQuart)])

    if len(voltage) > M:
        voltage = voltage[:M]
        time = time[:M]
    else:
        voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
        time = np.concatenate([time, np.zeros(M - len(time))])


    ax.plot(voltage*1e3, current.flatten()*1e6, **kwargs)
    return ax


def create_iv_plot(data_list):
    fig, ax = plt.subplots()
    colors = plt.cm.cividis(np.linspace(0, 1, int(len(data_list)/2)+1))
    for i, data in enumerate(data_list):
        heater_current = np.abs(data["heater_current"].flatten()[0]*1e6)
        ax = plot_iv_curve(data, ax, color=colors[i], zorder=-i, label=f"{heater_current:.0f} µA")
        if i == 10:
            break
    ax.set_ylim([-500, 500])
    ax.set_xlabel("Voltage [mV]")
    ax.set_ylabel("Current [µA]")
    ax.tick_params(direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(250))
    ax.yaxis.set_major_locator(plt.MultipleLocator(100))
    plt.legend()
    # plt.axis("equal")
    plt.savefig("iv_curve.pdf", bbox_inches="tight")
    plt.show()


def get_critical_currents(data_list):
    critical_currents = []
    critical_currents_std = []
    for data in data_list:
        heater_current = data["heater_current"]
        time = data["trace"][0,:]
        voltage = data["trace"][1,:]

        M = int(np.round(len(voltage), -2))
        currentQuart = np.linspace(0, data['vpp'] / 2 / 10e3, M // 4)
        current = np.concatenate([-currentQuart, np.flip(-currentQuart), currentQuart, np.flip(currentQuart)])

        if len(voltage) > M:
            voltage = voltage[:M]
            time = time[:M]
        else:
            voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
            time = np.concatenate([time, np.zeros(M - len(time))])

        current_time_trend = data["vpp"]/2/10e3*(data["time_trend"][1,:])/(1/(data["freq"]*4))*1e6
        avg_critical_current = np.mean(current_time_trend)
        std_critical_current = np.std(current_time_trend)
        critical_currents.append(avg_critical_current)
        critical_currents_std.append(std_critical_current)
    return critical_currents, critical_currents_std

def plot_critical_currents(data_list):
    critical_currents, critical_currents_std = get_critical_currents(data_list)
    fig, ax = plt.subplots()
    heater_currents = [data["heater_current"].flatten()*1e6 for data in data_list]
    ax.errorbar(heater_currents, critical_currents, yerr=critical_currents_std, fmt="o", markersize=3, color="darkblue")
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Heater Current [µA]")
    ax.set_ylabel("Critical Current [µA]")
    ax.set_ylim([0, 400])
    plt.savefig("critical_currents.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    data_list = import_directory(r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\dc_sweep")

    
    # create_iv_plot(data_list)

    critical_currents = get_critical_currents(data_list)
    
    plot_critical_currents(data_list)