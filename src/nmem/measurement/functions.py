# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:01:31 2023

@author: omedeiro
"""

from datetime import datetime
from time import sleep

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from qnnpy.functions.ntron import nTron
from scipy.optimize import curve_fit
from scipy.stats import norm
from tqdm import tqdm

from nmem.calculations.calculations import (
    htron_critical_current,
    calculate_heater_power,
)

# %% Functionss


def create_waveforms(
    num_samples: int = 256,
    width: int = 30,
    height: int = 1,
    phase: int = 0,
    ramp: bool = False,
) -> np.ndarray:
    """
    Create waveforms with specified parameters.

    Parameters:
        num_samples (int): The number of samples in the waveform. Default is 256.
        width (int): The width of the waveform. Default is 30.
        height (int): The height of the waveform. Default is 1.
        phase (int): The phase shift of the waveform. Default is 0.
        ramp (bool): Whether to create a ramp waveform. Default is False.

    Returns:
        numpy.ndarray: The generated waveform.

    """
    waveform = np.zeros(num_samples)

    middle = np.floor(num_samples / 2) + phase
    half = np.floor(width / 2)
    start = int(middle - half)
    stop = int(start + width)
    if stop > num_samples:
        stop = int(num_samples)

    if ramp:
        waveform[start:stop] = np.linspace(0, height, int(np.floor(width)))
    else:
        waveform[start:stop] = height

    return waveform


def write_waveforms(b: nTron, write_string: str, chan: int):
    name = "CHAN"
    sequence = f'"{name}",{write_string}'
    n = str(len(sequence))
    header = f"SOUR{chan}:DATA:SEQ #{len(n)}{n}"
    message = header + sequence
    # clears volatile waveform memory

    b.inst.awg.pyvisa.write_raw(message)
    b.inst.awg.write(f"SOURce{chan}:FUNCtion:ARBitrary {name}")
    # b.inst.awg.get_error()

    b.inst.awg.set_arb_sync()

    sleep(1)


def load_waveforms(
    b: nTron,
    measurement_settings: dict,
    chan: int = 1,
    ramp_read: bool = True,
):
    ww = measurement_settings["write_width"]
    rw = measurement_settings["read_width"]

    wr_ratio = (
        measurement_settings["write_current"] / measurement_settings["read_current"]
    )

    eww = measurement_settings["enable_write_width"]
    erw = measurement_settings["enable_read_width"]

    ew_phase = measurement_settings["enable_write_phase"]
    er_phase = measurement_settings["enable_read_phase"]

    ewr_ratio = (
        measurement_settings["enable_write_current"]
        / measurement_settings["enable_read_current"]
    )

    num_samples = measurement_settings["num_samples"]

    if wr_ratio >= 1:
        write_0 = create_waveforms(width=ww, height=-1)
        write_1 = create_waveforms(width=ww, height=1)
        read_wave = create_waveforms(width=rw, height=1 / wr_ratio, ramp=ramp_read)
    else:
        write_0 = create_waveforms(width=ww, height=-wr_ratio)
        write_1 = create_waveforms(width=ww, height=wr_ratio)
        read_wave = create_waveforms(width=rw, height=1, ramp=ramp_read)

    if ewr_ratio >= 1:
        enable_write = create_waveforms(width=eww, height=1, phase=ew_phase)
        enable_read = create_waveforms(width=erw, height=1 / ewr_ratio, phase=er_phase)
    else:
        enable_write = create_waveforms(width=eww, height=ewr_ratio, phase=ew_phase)
        enable_read = create_waveforms(width=erw, height=1, phase=er_phase)

    null_wave = create_waveforms(height=0)

    b.inst.awg.set_arb_wf(write_0, name="WRITE0", num_samples=num_samples, chan=chan)
    b.inst.awg.set_arb_wf(write_1, name="WRITE1", num_samples=num_samples, chan=chan)
    b.inst.awg.set_arb_wf(read_wave, name="READ", num_samples=num_samples, chan=chan)
    b.inst.awg.set_arb_wf(
        enable_write, name="ENABW", num_samples=num_samples, chan=chan
    )
    b.inst.awg.set_arb_wf(enable_read, name="ENABR", num_samples=num_samples, chan=chan)
    b.inst.awg.set_arb_wf(null_wave, name="WNULL", num_samples=num_samples, chan=chan)

    b.inst.awg.write(f"SOURce{chan}:DATA:VOLatile:CLEar")

    write1 = '"INT:\WRITE1.ARB"'
    write0 = '"INT:\WRITE0.ARB"'
    read = '"INT:\READ.ARB"'
    enabw = '"INT:\ENABW.ARB"'
    enabr = '"INT:\ENABR.ARB"'
    wnull = '"INT:\WNULL.ARB"'

    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {write0}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {write1}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {read}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {enabw}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {enabr}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {wnull}")


def voltage2current(v: float, c: float) -> float:
    if c == 1:
        current = v / 0.1 * 169.6 / 1e6
    if c == 2:
        current = v * (0.778 / 235.5)
    return current


def calculate_voltage(measurement_settings: dict) -> dict:
    enable_write_current = measurement_settings["enable_write_current"]
    write_current = measurement_settings["write_current"]

    read_current = measurement_settings["read_current"]
    enable_read_current = measurement_settings["enable_read_current"]

    enable_peak = max(enable_write_current, enable_read_current)
    enable_voltage = enable_peak / (0.778 / 235.5)
    channel_voltage = 0.1 / 169.6 * (read_current + write_current) * 1e6
    channel_voltage_read = 0.1 / 169.6 * (read_current) * 1e6

    measurement_settings["channel_voltage"] = channel_voltage
    measurement_settings["channel_voltage_read"] = channel_voltage_read
    measurement_settings["enable_voltage"] = enable_voltage

    if read_current == 0:
        wr_ratio = 0
    else:
        wr_ratio = write_current / read_current

    if enable_read_current == 0:
        ewr_ratio = 0
    else:
        ewr_ratio = enable_write_current / enable_read_current

    measurement_settings["wr_ratio"] = wr_ratio
    measurement_settings["ewr_ratio"] = ewr_ratio
    return measurement_settings


def gauss(x: float, mu: float, sigma: float, A: float):
    return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)


def bimodal(
    x: float, mu1: float, sigma1: float, A1: float, mu2: float, sigma2: float, A2: float
):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def bimodal_fit(
    x: np.ndarray,
    y: np.ndarray,
    expected: tuple,
    bounds: tuple = ((500, 2, 1e-6, 500, -30, 1e-6), (1200, 30, 1, 1200, 30, 1)),
) -> tuple:
    y = np.nan_to_num(y, posinf=0.0, neginf=0.0)
    params, cov = curve_fit(bimodal, x, y, expected, maxfev=5000, bounds=bounds)
    return params, cov


def get_param_mean(param: np.ndarray) -> np.ndarray:
    if round(param[2], 5) > round(param[5], 5):
        prm = param[0:2]
    else:
        prm = param[3:5]
    return prm


def reject_outliers(data: np.ndarray, m: float = 2.0):
    ind = abs(data - np.mean(data)) < m * np.std(data)
    if len(ind[ind is False]) < 50:
        data = data[ind]
        print(f"Samples rejected {len(ind[ind is False])}")
        rejectInd = np.invert(ind)
    else:
        rejectInd = None
    return data, rejectInd


def plot_histogram(
    ax: Axes, y: np.ndarray, label: str, color: str, previous_params: list = [0]
) -> tuple:
    y = y[y > 300]

    if len(previous_params) == 1:
        ax, param, full_params = plot_hist_bimodal(ax, y, label=f"{label}", color=color)

    else:
        # expected = (800, 8, 0.05, 1000, 8, 0.05)
        ax, param, full_params = plot_hist_bimodal(
            ax, y, label=f"{label}", color=color, expected=previous_params
        )
    return ax, param, full_params


def plot_hist_bimodal(
    ax: Axes,
    y: np.ndarray,
    label: str,
    color: str,
    expected: tuple = (700, 10, 0.01, 1040, 10, 0.05),
):
    binwidth = 4
    h, edges, _ = ax.hist(
        y,
        bins=range(300, 1200 + binwidth, binwidth),
        label=label,
        color=color,
        density=True,
    )
    binwidth = np.diff(edges).mean()
    edges = (edges[1:] + edges[:-1]) / 2  # for len(x)==len(y)
    x = np.linspace(edges[1], edges[-1], 2000)

    try:
        full_params, cov = bimodal_fit(edges, h, expected)
        ax.plot(x, bimodal(x, *full_params), color="k", linewidth=1)
        lbl = "b"
        params = get_param_mean(full_params)

    except Exception:
        try:
            full_params, cov = bimodal_fit(edges, h, (700, 10, 0.02, 1040, 10, 0.02))
            lbl = "b"
            ax.plot(x, bimodal(x, *full_params), color="k", linewidth=1)
            lbl = "b"
            params = get_param_mean(full_params)

        except Exception:
            full_params, cov = bimodal_fit(edges, h, (700, 10, 0.02, 750, 10, 0.02))
            lbl = "b"
            params = get_param_mean(full_params)

    if (abs(np.array([full_params[2], full_params[5]])) < 1e-5).any() is True:
        try:
            ax.plot(x, bimodal(x, *params), color="k", linewidth=1)
            full_params = params
            params = get_param_mean(params)
            lbl = "b"
        except Exception:
            params = norm.fit(y)
            pdf = norm.pdf(x, *params) * h.sum() * binwidth
            ax.plot(x, pdf, color="k", linewidth=1)
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
            lbl = "g"

    if (np.array([full_params[1], full_params[4]]) < 3).any() is True:
        try:
            params = norm.fit(y)
            expected = np.array([params[0] - 20, 12, 0.05, params[0] + 20, 10, 0.05])
            params, cov = bimodal_fit(edges, h, expected)
            ax.plot(x, bimodal(x, *params), color="k", linewidth=1)
            lbl = "b"
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
        except Exception:
            params = norm.fit(y)
            pdf = norm.pdf(x, *params) * h.sum() * binwidth
            ax.plot(x, pdf, color="k", linewidth=1)
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
            lbl = "g"

    ax.plot(
        [params[0] + abs(params[1]), params[0] + abs(params[1])],
        [0, 0.1],
        color="k",
        linewidth=0.5,
        label=f"{lbl}{abs(params[1]):.2f}",
    )

    ax.plot(
        [params[0] - abs(params[1]), params[0] - abs(params[1])],
        [0, 0.1],
        color="k",
        linewidth=0.5,
    )

    # ax.plot([full_params[0], full_params[0]], [0, 0.1], color='b', linewidth=0.5)
    # ax.plot([full_params[3], full_params[3]], [0, 0.1], color='b', linewidth=0.5)

    return ax, params, full_params


# def plot_hist(ax, y, label, color):
#     binwidth=3
#     h, edges, _ = ax.hist(y, bins=range(500, 1200 + binwidth, binwidth), color=color,  density = True)
#     param = norm.fit(y)   # Fit a normal distribution to the data
#     binwidth = np.diff(edges).mean()
#     edges = (edges[1:]+edges[:-1])/2 # for len(x)==len(y)
#     x = np.linspace(edges[1], edges[-1], 2000)
#     pdf = norm.pdf(x, *param)*h.sum()*binwidth
#     ax.plot(x, pdf, color = 'g', linewidth=1)
#     return ax, param


def plot_waveforms(
    data_dict: dict, measurement_settings: dict, previous_params: list = [0]
):
    i0 = data_dict["i0"]
    i1 = data_dict["i1"]
    bitmsg_channel = measurement_settings["bitmsg_channel"]
    bitmsg_enable = measurement_settings["bitmsg_enable"]

    traces = 1
    if traces == 1:
        trace_chan_in = data_dict["trace_chan_in"]
        trace_chan_out = data_dict["trace_chan_out"]
        trace_enab = data_dict["trace_enab"]

        numpoints = int((len(trace_chan_in[1]) - 1) / 2)

        ax0 = plt.subplot(411)
        ax0.plot(
            trace_chan_in[0] * 1e6,
            trace_chan_in[1] * 1e3 / 50 - 6,
            label=f"peak current = {max(trace_chan_in[1][0:1000]*1e3/50):.1f}mA peak voltage = {max(trace_chan_in[1][0:1000]*1e3):.1f}mV",
            color="C7",
        )
        ax0.legend(loc=3)
        plt.ylim((-12, 12))
        plt.ylabel("splitter current (mA)")

        half = False
        if half is True:
            ax1 = plt.subplot(412)
            ax1.plot(
                (
                    trace_chan_out[0][numpoints + 1 : numpoints * 2 + 1]
                    - trace_chan_out[0][numpoints]
                )
                * 1e6,
                trace_chan_out[1][numpoints + 1 : numpoints * 2 + 1] * 1e3,
                label="WRITE 0",
                color="C0",
            )
            ax1.plot(
                trace_chan_out[0][0:numpoints] * 1e6,
                trace_chan_out[1][0:numpoints] * 1e3,
                label="WRITE 1",
                color="C2",
            )
        else:
            ax1 = ax0.twinx()
            ax1.plot(
                trace_chan_out[0] * 1e6,
                trace_chan_out[1] * 1e3,
                label="channel",
                color="C4",
            )
            plt.ylim((-700, 700))
            plt.ylabel("volage (mV)")
            ax1.legend(loc=4)

        ax1a = plt.subplot(412)
        ax1a.plot(
            np.array([trace_chan_out[0][0], trace_chan_out[0][-1]]) * 1e6,
            np.tile(measurement_settings["threshold_read"], 2) * 1e3,
            linestyle="dashed",
            color="C0",
        )
        # print(max(data3[0]))
        ax1a.legend(loc=2)

        plt.xlabel("time (us)")
        plt.ylabel("volage (mV)")
        plt.ylim((-700, 700))
        plt.xlim((0, 10))

        ax2 = ax1a.twinx()
        if half is True:
            ax2.plot(
                trace_enab[0][0:numpoints] * 1e6,
                trace_enab[1][0:numpoints] * 1e3,
                label="enable",
                color="C1",
            )
            ax2.plot(
                (
                    trace_enab[0][numpoints + 1 : numpoints * 2 + 1]
                    - trace_enab[0][numpoints]
                )
                * 1e6,
                trace_enab[1][numpoints + 1 : numpoints * 2 + 1] * 1e3,
                label="enable1",
                color="C1",
            )
        else:
            ax2.plot(
                trace_enab[0] * 1e6, trace_enab[1] * 1e3, label="enable", color="C1"
            )

        ax2.plot(
            np.array([trace_chan_out[0][0], trace_chan_out[0][-1]]) * 1e6,
            np.tile(measurement_settings["threshold_enab"], 2) * 1e3,
            linestyle="dashed",
            color="C0",
        )
        plt.ylim((0, 300))
        ax2.legend(loc=1)
        plt.ylabel("volage (mV)")
        plt.xlim((0, 10))

    ax3 = plt.subplot(413)
    y0 = i0[i0 > 0] * 1e6
    y0 = y0[y0 < 1100]

    y1 = i1[i1 > 0] * 1e6
    y1 = y1[y1 < 1100]

    if len(previous_params) > 1:
        pparams0 = previous_params[0]
        pparams1 = previous_params[1]

    else:
        pparams0 = [0]
        pparams1 = [0]

    ax3, param0, full_params0 = plot_histogram(
        ax3, y0, label="READ 0", color="C0", previous_params=pparams0
    )
    ax3, param1, full_params1 = plot_histogram(
        ax3, y1, label="READ 1", color="C2", previous_params=pparams1
    )
    plt.xlim([300, 1200])
    plt.ylim([-0.01, 0.08])

    distance = param0[0] - param1[0]
    # ydiff = pdf0-pdf1
    # np.mean([pdf0[ydiff==min(ydiff)],    pdf1[ydiff==min(ydiff)]])
    ber_est = 0

    plt.xlabel("read current (uA)")
    plt.ylabel("pdf")
    # plt.xlim([150, 350])
    ax3.legend(loc=2)

    ax4 = plt.subplot(414)
    ax4 = plot_trend(ax4, i0[i0 > 0] * 1e6, label="READ 0", color="C0", params=param0)
    ax4 = plot_trend(ax4, i1[i1 > 0] * 1e6, label="READ 1", color="C2", params=param1)
    plt.ylim([300, 1200])
    plt.ylabel("read current (uA)")
    plt.xlabel("sample")

    channel_voltage = measurement_settings["channel_voltage"]
    enable_voltage = measurement_settings["enable_voltage"]
    read_current = measurement_settings["read_current"]
    write_current = measurement_settings["write_current"]
    enable_read_current = measurement_settings["enable_read_current"]
    enable_write_current = measurement_settings["enable_write_current"]

    plt.suptitle(
        f"Write Current:{write_current*1e6:.1f}uA, Read Current:{read_current*1e6:.0f}uA, "
        "\n enable_write_current:{enable_write_current*1e6:.1f}uA, enable_read_current:{enable_read_current*1e6:.1f}uA"
        "\n Vcpp:{channel_voltage*1e3:.1f}mV, Vepp:{enable_voltage*1e3:.1f}mV, Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}"
    )

    plt.tight_layout()

    saves = 0
    if saves == 1:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(f"figure_{timestamp}.png")

    plt.show()

    return distance, full_params0, full_params1


def plot_trend(
    ax: Axes,
    y: np.ndarray,
    label: str,
    color: str,
    x: np.ndarray = None,
    params: tuple = None,
    m: float = 1,
):
    if x is None:
        x = np.arange(0, len(y), 1)

    ax.plot(x, y, ls="none", marker="o", label=label, color=color)
    if len(x) > 1:
        if params is not None:
            ax.plot(
                [x[0], x[-1]],
                np.tile(params[0] + params[1] * m, 2),
                color=np.subtract(1, mpl.colors.to_rgb(color)),
                linewidth=2,
                label=f"{label}",
            )
            ax.plot(
                [x[0], x[-1]],
                np.tile(params[0] - params[1] * m, 2),
                color=np.subtract(1, mpl.colors.to_rgb(color)),
                linewidth=2,
            )

    # if ind is not None:
    #     ax.plot(x[ind], y[ind], ls='none', marker='x', markersize=15, color=color)
    # print(ind)

    return ax


# def plot_waveforms_lite(data0, data1, data2, i0, i1, bitmsg_channel='1R0R', bitmsg_enable = 'WRWR'):
#     numpoints = int((len(data0[1])-1)/2)
#     ax3 = plt.axes()

#     ax3.hist(i0[i0>0]*1e6, 30, label='READ 0 ', color='C0')
#     ax3.hist(i1[i1>0]*1e6, 30, label='READ 1', color='C2')
#     plt.xlabel('read current (uA)')
#     plt.ylabel('count')
#     # plt.xlim([200, 1000])
#     ax3.legend(loc=2)
#     plt.suptitle(
#          f'Vpp:{channel_voltage*1e3:.1f}mV, Read Current:{read_current*1e6:.0f}uA, Write Current:{write_current*1e6:.1f}, enable_current:{enable_read_current*1e6:.1f} \n Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}')

#     plt.show()


def plot_ber(x: np.ndarray, y: np.ndarray, ber: np.ndarray):
    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]

    cmap = plt.get_cmap("RdBu", 100).reversed()
    plt.imshow(
        np.reshape(ber, (len(x), len(y))),
        extent=[
            (-0.5 * dx + xstart),
            (0.5 * dx + xstop),
            (-0.5 * dy + ystart),
            (0.5 * dy + ystop),
        ],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    # plt.xticks(np.arange(xstart*1e3,xstop*1e3, len(x)), rotation=45)
    # plt.yticks(np.arange(ystart*1e3,ystop*1e3, len(y)))
    plt.colorbar()


def plot_waveforms_bert(data_dict: dict, measurement_settings: dict):
    trace_chan_in = data_dict["trace_chan_in"]
    trace_chan_out = data_dict["trace_chan_out"]
    trace_enab = data_dict["trace_enab"]
    read_zero_top = data_dict["read_zero_top"]
    read_one_top = data_dict["read_one_top"]

    channel_voltage = measurement_settings["channel_voltage"]
    enable_voltage = measurement_settings["enable_voltage"]
    read_current = measurement_settings["read_current"]
    write_current = measurement_settings["write_current"]
    enable_read_current = measurement_settings["enable_read_current"]
    enable_write_current = measurement_settings["enable_write_current"]
    bitmsg_channel = measurement_settings["bitmsg_channel"]
    bitmsg_enable = measurement_settings["bitmsg_enable"]
    measurement_name = measurement_settings["measurement_name"]
    sample_name = measurement_settings["sample_name"]

    numpoints = int((len(trace_chan_in[1]) - 1) / 2)

    ax0 = plt.subplot(411)
    ax0.plot(
        trace_chan_in[0] * 1e6,
        trace_chan_in[1] * 1e3 / 50,
        label=f"peak current = {max(trace_chan_in[1][0:1000]*1e3/50):.1f}mA peak voltage = {max(trace_chan_in[1][0:1000]*1e3):.1f}mV",
        color="C7",
    )
    ax0.legend(loc=1)
    plt.ylabel("splitter current (mA)")

    ax1 = plt.subplot(412)
    ax1.plot(
        trace_chan_out[0] * 1e6, trace_chan_out[1] * 1e3, label="channel", color="C4"
    )
    plt.xlabel("time (us)")
    plt.ylabel("volage (mV)")
    ax1.legend(loc=1)
    plt.xlim((0, measurement_settings["horizontal_scale"] * 10 * 1e6))

    ax2 = plt.subplot(413)
    ax2.plot(trace_enab[0] * 1e6, trace_enab[1] * 1e3, label="enable", color="C1")
    ax2.legend(loc=1)
    plt.xlabel("time (us)")
    plt.ylabel("volage (mV)")
    plt.xlim((0, measurement_settings["horizontal_scale"] * 10 * 1e6))

    ax3 = plt.subplot(414)
    ax3 = plot_trend(ax3, read_zero_top, label="READ 0", color="C0")
    ax3 = plot_trend(ax3, read_one_top, label="READ 1", color="C2")
    ax3.hlines(
        measurement_settings["threshold_bert"], 0, len(read_zero_top), color="C0"
    )
    plt.ylim([0, 0.5])
    # plt.ylabel('read current (uA)')
    plt.xlabel("sample")
    ax3.legend()

    plt.suptitle(
        f"{sample_name} -- {measurement_name} \n Vcpp:{channel_voltage*1e3:.1f}mV, Vepp:{enable_voltage*1e3:.1f}mV, Write Current:{write_current*1e6:.1f}, Read Current:{read_current*1e6:.0f}uA, \n enable_write_current:{enable_write_current*1e6:.1f}, enable_read_current:{enable_read_current*1e6:.1f} \n Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}"
    )

    plt.tight_layout()

    plt.show()


def plot_parameter(
    data_dict: dict,
    x_name: str,
    y_name: str,
    xindex: int = 0,
    yindex: int = 0,
    ax: Axes = None,
    **kwargs,
):
    x_length = data_dict["x"].shape[1]
    y_length = data_dict["y"].shape[1]

    if not ax:
        fig, ax = plt.subplots()

    x = data_dict[x_name][xindex].flatten().reshape(x_length, y_length).squeeze()
    y = data_dict[y_name][yindex].flatten().reshape(x_length, y_length)
    ax.plot(x.flatten(), y.flatten(), marker="o", label=y_name, **kwargs)
    plt.xticks(rotation=60)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    measurement_name = data_dict["measurement_name"].flatten()
    sample_name = data_dict["sample_name"].flatten()

    time_str = data_dict["time_str"]

    channel_voltage = data_dict["channel_voltage"].flatten()
    enable_voltage = data_dict["enable_voltage"].flatten()
    read_current = data_dict["read_current"].flatten()
    write_current = data_dict["write_current"].flatten()
    enable_read_current = data_dict["enable_read_current"].flatten()
    enable_write_current = data_dict["enable_write_current"].flatten()
    bitmsg_channel = data_dict["bitmsg_channel"].flatten()
    bitmsg_enable = data_dict["bitmsg_enable"].flatten()

    plt.suptitle(
        f"{sample_name[0]} -- {measurement_name[0]} \n {time_str} \n Vcpp:{channel_voltage[0]*1e3:.1f}mV, Vepp:{enable_voltage[0]*1e3:.1f}mV, Write Current:{write_current[0]*1e6:.1f}uA, Read Current:{read_current[0]*1e6:.0f}uA, \n enable_write_current:{enable_write_current[0]*1e6:.1f}, enable_read_current:{enable_read_current[0]*1e6:.1f} \n Channel_message: {bitmsg_channel[0]}, Channel_enable: {bitmsg_enable[0]}"
    )

    # plt.suptitle(
    #     f'{sample_name[0]} -- {measurement_name[0]} \n {time_str}')

    print(f"min {y_name} at {x_name} = {x[y.argmin()]}")
    return ax


def plot_array(
    data_dict: dict,
    c_name: str,
    x_name: str = "x",
    y_name: str = "y",
    ax: Axes = None,
    cmap=None,
    norm=True,
):
    if not ax:
        fig, ax = plt.subplots()

    x_length = data_dict[x_name].shape[1]
    y_length = data_dict[y_name].shape[1]

    x = data_dict["x"][0][:, 0] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    c = data_dict[c_name][0].flatten()

    ctotal = c.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]

    if not cmap:
        cmap = plt.get_cmap("RdBu", 100).reversed()

    plt.imshow(
        ctotal,
        extent=[
            (-0.5 * dx + xstart),
            (0.5 * dx + xstop),
            (-0.5 * dy + ystart),
            (0.5 * dy + ystop),
        ],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )

    # print(f'{dx}, {xstart}, {xstop}, {dy}, {ystart}, {ystop}')
    plt.xticks(np.linspace(xstart, xstop, len(x)), rotation=45)
    plt.yticks(np.linspace(ystart, ystop, len(y)))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    xv, yv = np.meshgrid(x, y, indexing="ij")

    if norm:
        plt.clim([0, 1])
        cbar = plt.colorbar(ticks=np.linspace(0, 1, 11))
    else:
        plt.clim([0, c.max()])
        cbar = plt.colorbar()
    cbar.ax.set_ylabel(c_name, rotation=270)
    plt.contour(xv, yv, np.reshape(ctotal, (len(y), len(x))).T, [0.05, 0.1])
    measurement_name = data_dict["measurement_name"].flatten()
    sample_name = data_dict["sample_name"].flatten()
    time_str = data_dict["time_str"]

    plt.suptitle(f"{sample_name[0]} -- {measurement_name[0]} \n {time_str}")

    return ax


def get_traces(b: nTron, scope_samples: int = 5000):
    # b.inst.scope.set_sample_mode('Realtime')
    sleep(1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(0.1)
    trace_chan_in: np.ndarray = b.inst.scope.get_wf_data("C1")
    sleep(0.1)
    trace_chan_out: np.ndarray = b.inst.scope.get_wf_data("C2")
    sleep(0.1)
    trace_enab: np.ndarray = b.inst.scope.get_wf_data("C4")

    trace_chan_in = np.resize(trace_chan_in, (2, scope_samples))
    trace_chan_out = np.resize(trace_chan_out, (2, scope_samples))
    trace_enab = np.resize(trace_enab, (2, scope_samples))

    b.inst.scope.set_trigger_mode("Normal")
    sleep(1e-2)

    try:
        time_est = round(b.inst.scope.get_parameter_value("P2"), 8)
        b.inst.scope.set_math_vertical_scale("F1", 50e-9, time_est)
        b.inst.scope.set_math_vertical_scale("F2", 50e-9, time_est)
    except Exception:
        sleep(1e-4)

    return (
        trace_chan_in,
        trace_chan_out,
        trace_enab,
    )


def get_traces_sequence(b: nTron, num_meas: int = 100, num_samples: int = 5000):
    data_dict = []
    for c in ["C1", "C2", "C4", "F4"]:
        x, y = b.inst.scope.get_wf_data(channel=c)
        interval = abs(x[0] - x[1])
        xlist = []
        ylist = []
        totdp = np.int64(np.size(x) / num_meas)

        for j in range(1):
            xx = x[0 + j * totdp : totdp + j * totdp] - totdp * interval * j
            yy = y[0 + j * totdp : totdp + j * totdp]

            xlist.append(xx[0 : int(num_samples)])
            ylist.append(yy[0 : int(num_samples)])

        trace_dict = [xlist, ylist]
        data_dict.append(trace_dict)

    data0 = data_dict[0]
    data1 = data_dict[1]
    data2 = data_dict[2]
    data3 = data_dict[3]

    data0 = np.resize(data0, (2, num_samples))
    data1 = np.resize(data1, (2, num_samples))
    data2 = np.resize(data2, (2, num_samples))
    data3 = np.resize(data3, (2, num_samples))

    try:
        time_est = round(b.inst.scope.get_parameter_value("P2"), 8)
        b.inst.scope.set_math_vertical_scale("F1", 50e-9, time_est)
        b.inst.scope.set_math_vertical_scale("F2", 50e-9, time_est)
    except Exception:
        sleep(1e-4)

    return data0, data1, data2, data3


def run_realtime(b: nTron, num_meas: int = 100):
    b.inst.scope.set_trigger_mode("Normal")
    sleep(0.5)
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    pbar = tqdm(total=num_meas)
    while b.inst.scope.get_num_sweeps() < num_meas:
        sleep(0.1)
        n = b.inst.scope.get_num_sweeps()
        pbar.update(n - pbar.n)

    b.inst.scope.set_trigger_mode("Stop")

    t1 = b.inst.scope.get_wf_data("F1")
    t0 = b.inst.scope.get_wf_data("F2")
    pbar.close()
    return t0, t1


def run_realtime_bert(b: nTron, measurement_settings: dict):
    if measurement_settings["num_meas"]:
        num_meas = measurement_settings["num_meas"]
    else:
        num_meas = 100

    if measurement_settings["threshold_bert"]:
        threshold = measurement_settings["threshold_bert"]
    else:
        threshold = 150e-3

    b.inst.scope.set_trigger_mode("Normal")
    sleep(0.5)
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    with tqdm(total=num_meas) as pbar:
        while b.inst.scope.get_num_sweeps() < num_meas:
            sleep(0.1)
            n = b.inst.scope.get_num_sweeps()
            pbar.update(n - pbar.n)
            # print(f"sampling...{n} / {num_meas}")

    b.inst.scope.set_trigger_mode("Stop")

    # This assumes that the measurements are always zero then one.
    read_zero_top = b.inst.scope.get_wf_data("F5")
    read_one_top = b.inst.scope.get_wf_data("F6")

    read_zero_top = read_zero_top[1][0:num_meas]
    read_one_top = read_one_top[1][0:num_meas]

    read_zero_top = read_zero_top.flatten()
    read_one_top = read_one_top.flatten()

    if len(read_zero_top) < num_meas:
        read_zero_top.resize(num_meas, refcheck=False)
    if len(read_one_top) < num_meas:
        read_one_top.resize(num_meas, refcheck=False)

    # READ 1: below threshold (no voltage)
    write_0_read_1 = (read_zero_top < threshold).sum()

    # READ 0: above threshold (voltage)
    write_1_read_0 = (read_one_top > threshold).sum()

    write_0_read_1_norm = write_0_read_1 / (num_meas * 2)
    write_1_read_0_norm = write_1_read_0 / (num_meas * 2)

    return (
        read_zero_top,
        read_one_top,
        write_0_read_1,
        write_1_read_0,
        write_0_read_1_norm,
        write_1_read_0_norm,
    )


def run_sequence_bert(b: nTron, measurement_settings: dict):
    if measurement_settings["num_meas"]:
        num_meas = measurement_settings["num_meas"]
    else:
        num_meas = 100

    b.inst.scope.clear_sweeps()
    sleep(0.1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(1)
    while b.inst.scope.get_trigger_mode() != "Stopped\n":
        sleep(1)
        print("sampling...")

    read_zero_top = b.inst.scope.get_wf_data("F5")
    read_one_top = b.inst.scope.get_wf_data("F6")

    read_zero_top = read_zero_top[1][0:num_meas]
    read_one_top = read_one_top[1][0:num_meas]

    read_zero_top = read_zero_top.flatten()
    read_one_top = read_one_top.flatten()

    if len(read_zero_top) < num_meas:
        read_zero_top.resize(num_meas, refcheck=False)
    if len(read_one_top) < num_meas:
        read_one_top.resize(num_meas, refcheck=False)

    if measurement_settings["threshold_bert"]:
        threshold = measurement_settings["threshold_bert"]
    else:
        threshold = 150e-3
    # READ 1: below threshold (no voltage)
    write_0_read_1 = (read_zero_top < threshold).sum()

    # READ 0: above threshold (voltage)
    write_1_read_0 = (read_one_top > threshold).sum()

    write_0_read_1_norm = write_0_read_1 / (num_meas * 2)
    write_1_read_0_norm = write_1_read_0 / (num_meas * 2)

    print(f"w0r1: {write_0_read_1}, w1r0: {write_1_read_0}")
    return (
        read_zero_top,
        read_one_top,
        write_0_read_1,
        write_1_read_0,
        write_0_read_1_norm,
        write_1_read_0_norm,
    )


def run_sequence(b: nTron, num_meas: int = 100, num_samples: int = 5e3):
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(1)
    while b.inst.scope.get_trigger_mode() != "Stopped\n":
        sleep(1)
        print("sampling...")

    t1 = b.inst.scope.get_wf_data("F1")
    t0 = b.inst.scope.get_wf_data("F2")

    save_full_dict = 0
    if save_full_dict == 1:
        full_dict = {}
        for c in ["C1", "C2", "C3", "C4"]:
            x, y = b.inst.scope.get_wf_data(channel=c)
            interval = abs(x[0] - x[1])
            xlist = []
            ylist = []
            totdp = np.int64(np.size(x) / num_meas)

            for j in range(num_meas):
                xx = x[0 + j * totdp : totdp + j * totdp] - totdp * interval * j
                yy = y[0 + j * totdp : totdp + j * totdp]

                xlist.append(xx[0 : int(num_samples)])
                ylist.append(yy[0 : int(num_samples)])

            data_dict = {c + "x": xlist, c + "y": ylist}
            full_dict.update(data_dict)
    else:
        full_dict = {}

    return t0, t1, full_dict


def run_sequence_v1(b: nTron, num_meas: int = 100):
    b.inst.scope.set_trigger_mode("Stop")
    sleep(0.1)

    sleep(0.1)
    b.inst.scope.set_segments(50)
    sleep(0.5)
    b.inst.scope.set_trigger_mode("Normal")
    sleep(1)
    b.inst.scope.clear_sweeps()
    while b.inst.scope.get_num_sweeps() < num_meas:
        sleep(0.1)
        n = b.inst.scope.get_num_sweeps()
        print(f"sampling...{n} / {num_meas}")

    t0 = b.inst.scope.get_wf_data("F1")

    t1 = b.inst.scope.get_wf_data("F2")

    return t0, t1


def calculate_currents(
    time_zero: np.ndarray,
    time_one: np.ndarray,
    measurement_settings: dict,
    total_points: int,
    sample_time: float,
):
    num_meas = measurement_settings["num_meas"]
    read_current = measurement_settings["read_current"]

    time_one = time_one[1][0:num_meas]
    time_zero = time_zero[1][0:num_meas]

    time_one = time_one.flatten()
    time_zero = time_zero.flatten()

    if len(time_zero) < num_meas:
        time_zero.resize(num_meas, refcheck=False)
    if len(time_one) < num_meas:
        time_one.resize(num_meas, refcheck=False)

    read_time = (measurement_settings["read_width"] / total_points) * sample_time

    current_zero = time_zero / read_time * read_current
    current_one = time_one / read_time * read_current

    mean0, std0 = norm.fit(current_zero * 1e6)
    mean1, std1 = norm.fit(current_one * 1e6)

    distance = mean0 - mean1  # in microamps

    if len(current_zero) != 0 and len(current_one) != 0:
        x = np.linspace(mean0, mean1, 100)

        y0 = norm.pdf(x, mean0, std0)

        y1 = norm.pdf(x, mean1, std1)

        ydiff = np.subtract(y0, y1)

    return time_zero, time_one, current_zero, current_one, distance, x, y0, y1


def calculate_error_rate(t0: np.ndarray, t1: np.ndarray, num_meas: int):
    w0r1 = len(np.argwhere(t0 > 0))
    w1r0 = num_meas - len(np.argwhere(t1 > 0))

    ber = (w0r1 + w1r0) / (2 * num_meas)
    return ber, w0r1, w1r0


def write_sequence(
    b: nTron,
    message: str,
    channel: int,
    measurement_settings: dict,
    ramp_read: bool = True,
):
    write1 = '"INT:\WRITE1.ARB"'
    write0 = '"INT:\WRITE0.ARB"'
    read = '"INT:\READ.ARB"'
    enabw = '"INT:\ENABW.ARB"'
    enabr = '"INT:\ENABR.ARB"'
    wnull = '"INT:\WNULL.ARB"'

    write_string = []
    start_flag = True
    for c in message:
        if start_flag is True:
            suffix = ",0,once,highAtStartGoLow,50"
        else:
            suffix = ",0,once,maintain,50"

        if c == "0":
            write_string.append(write0 + suffix)
        if c == "1":
            write_string.append(write1 + suffix)
        if c == "W":
            write_string.append(enabw + suffix)
        if c == "N":
            write_string.append(wnull + suffix)
        if c == "E":
            write_string.append(enabr + suffix)
        if c == "R":
            write_string.append(read + suffix)

        start_flag = False

    write_string = ",".join(write_string)
    # print(write_string)

    load_waveforms(b, measurement_settings, chan=channel, ramp_read=ramp_read)
    write_waveforms(b, write_string, channel)


def setup_scope(
    b: nTron,
    sample_time: float,
    horscale: float,
    scope_sample_rate: float,
    measurement_settings: dict,
):
    b.inst.scope.set_deskew("F3", min(sample_time / 200, 5e-6))

    b.inst.scope.set_horizontal_scale(horscale, -horscale * 5)
    b.inst.scope.set_sample_rate(max(scope_sample_rate, 1e6))

    num_meas = measurement_settings["num_meas"]
    b.inst.scope.set_measurement_clock_level(
        "P1", "1", "Absolute", measurement_settings["threshold_enab"]
    )
    b.inst.scope.set_measurement_clock_level(
        "P2", "1", "Absolute", measurement_settings["threshold_enab"]
    )

    b.inst.scope.set_measurement_clock_level(
        "P1", "2", "Absolute", measurement_settings["threshold_read"]
    )
    b.inst.scope.set_measurement_clock_level(
        "P2", "2", "Absolute", measurement_settings["threshold_read"]
    )

    div1 = 4.5
    div2 = 8.5

    b.inst.scope.set_measurement_gate("P1", div1 - 0.5, div1 + 0.85)
    b.inst.scope.set_measurement_gate("P2", div2 - 0.5, div2 + 0.85)

    b.inst.scope.set_math_trend_values("F1", num_meas * 2)
    b.inst.scope.set_math_trend_values("F2", num_meas * 2)


def setup_scope_bert(
    b: nTron,
    measurement_settings: dict,
):
    horizontal_scale = measurement_settings["horizontal_scale"]
    sample_time = measurement_settings["sample_time"]
    scope_sample_rate = measurement_settings["scope_sample_rate"]

    b.inst.scope.set_deskew("F3", min(sample_time / 200, 5e-6))

    b.inst.scope.set_horizontal_scale(horizontal_scale, -horizontal_scale * 5)
    b.inst.scope.set_sample_rate(max(scope_sample_rate, 1e6))

    num_meas = measurement_settings["num_meas"]
    b.inst.scope.set_measurement_clock_level(
        "P1", "1", "Absolute", measurement_settings["threshold_enab"]
    )
    b.inst.scope.set_measurement_clock_level(
        "P2", "1", "Absolute", measurement_settings["threshold_enab"]
    )

    b.inst.scope.set_measurement_clock_level(
        "P1", "2", "Absolute", measurement_settings["threshold_read"]
    )
    b.inst.scope.set_measurement_clock_level(
        "P2", "2", "Absolute", measurement_settings["threshold_read"]
    )

    div1 = 4.5
    div2 = 8.5

    b.inst.scope.set_measurement_gate("P3", div1 + 0.1, div1 + 0.3)
    b.inst.scope.set_measurement_gate("P4", div2 + 0.1, div2 + 0.3)

    b.inst.scope.set_math_trend_values("F5", num_meas * 2)
    b.inst.scope.set_math_trend_values("F6", num_meas * 2)
    b.inst.scope.set_math_vertical_scale("F5", 100e-3, 300e-3)
    b.inst.scope.set_math_vertical_scale("F6", 100e-3, 300e-3)


def run_measurement(
    b: nTron,
    measurement_settings: dict,
    save_traces: bool = False,
    plot: bool = False,
    ramp_read: bool = True,
):
    bitmsg_channel: str = measurement_settings["bitmsg_channel"]
    bitmsg_enable: str = measurement_settings["bitmsg_enable"]
    total_points: int = measurement_settings["num_samples"] * len(bitmsg_channel)
    sample_rate: float = measurement_settings["sample_rate"]
    horscale: float = measurement_settings["horizontal_scale"]
    sample_time: float = measurement_settings["sample_time"]
    channel_voltage: float = measurement_settings["channel_voltage"]
    enable_voltage: float = measurement_settings["enable_voltage"]
    scope_samples: int = int(measurement_settings["num_samples_scope"])
    scope_sample_rate = scope_samples / sample_time

    num_meas: int = measurement_settings["num_meas"]

    setup_scope_bert(b, measurement_settings)

    ######################################################

    if enable_voltage > 200e-3:
        raise ValueError("enable voltage too high")

    if channel_voltage > 1.5:
        raise ValueError("channel voltage too high")

    if channel_voltage == 0:
        bitmsg_channel = "N" * len(bitmsg_channel)
    if enable_voltage == 0:
        bitmsg_enable = "N" * len(bitmsg_enable)

    write_sequence(b, bitmsg_channel, 1, measurement_settings, ramp_read=ramp_read)
    b.inst.awg.set_vpp(channel_voltage, 1)
    b.inst.awg.set_arb_sample_rate(sample_rate, 1)

    write_sequence(b, bitmsg_enable, 2, measurement_settings, ramp_read=ramp_read)
    b.inst.awg.set_vpp(enable_voltage, 2)
    b.inst.awg.set_arb_sample_rate(sample_rate, 2)

    b.inst.awg.set_arb_sync()
    sleep(1)

    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)

    b.inst.scope.clear_sweeps()

    (
        read_zero_top,
        read_one_top,
        write_0_read_1,
        write_1_read_0,
        write_0_read_1_norm,
        write_1_read_0_norm,
    ) = run_realtime_bert(b, measurement_settings)

    trace_chan_in, trace_chan_out, trace_enab = get_traces(b, scope_samples)

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    bit_error_rate = (write_0_read_1 + write_1_read_0) / (2 * num_meas)
    DATA_DICT = {
        "trace_chan_in": trace_chan_in,
        "trace_chan_out": trace_chan_out,
        "trace_enab": trace_enab,
        "write_0_read_1": write_0_read_1,
        "write_1_read_0": write_1_read_0,
        "write_0_read_1_norm": write_0_read_1_norm,
        "write_1_read_0_norm": write_1_read_0_norm,
        "read_zero_top": read_zero_top,
        "read_one_top": read_one_top,
        "bit_error_rate": bit_error_rate,
    }

    if plot is True:
        plot_waveforms_bert(DATA_DICT, measurement_settings)

    return DATA_DICT


def update_dict(dict1: dict, dict2: dict):
    result_dict = {}
    for key in dict1.keys():
        try:
            result_dict[key] = np.dstack([dict1[key], dict2[key]])
        except Exception:
            print(f"could not stack {key}")

    return result_dict


def write_dict_to_file(file_path: str, save_dict: dict):
    with open(f"{file_path}_measurement_settings.txt", "w") as file:
        for key, value in save_dict.items():
            file.write(f"{key}: {value}\n")


def run_sweep(
    b: nTron,
    measurement_settings: dict,
    parameter_x: str,
    parameter_y: str,
    save_traces: bool = False,
    plot_measurement=False,
):
    save_dict = {}
    for x in measurement_settings["x"]:
        for y in measurement_settings["y"]:
            measurement_settings[parameter_x] = x
            measurement_settings[parameter_y] = y
            print(f"{parameter_x}: {x:.2e}, {parameter_y}: {y:.2e}")

            cell = b.properties["Save File"]["cell"]
            slope = measurement_settings["CELLS"][cell]["slope"]
            intercept = measurement_settings["CELLS"][cell]["intercept"]
            write_critical_current = htron_critical_current(
                measurement_settings["enable_write_current"] * 1e6, slope, intercept
            )
            read_critical_current = htron_critical_current(
                measurement_settings["enable_read_current"] * 1e6, slope, intercept
            )
            # print(f"Write Current: {measurement_settings['write_current']:.2f}")
            # print(f"Write Critical Current: {write_critical_current:.2f}")
            # print(f"Read Current: {measurement_settings['read_current']:.2f}")
            # print(f"Read Critical Current: {read_critical_current:.2f}")

            measurement_settings = calculate_voltage(measurement_settings)

            data_dict = run_measurement(
                b,
                measurement_settings,
                save_traces,
                plot=plot_measurement,
                ramp_read=False,
            )

            data_dict.update(measurement_settings)
            enable_write_power = calculate_heater_power(
                measurement_settings["enable_write_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            enable_read_power = calculate_heater_power(
                measurement_settings["enable_read_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            param_dict = {
                "Write Current": [measurement_settings["write_current"] * 1e6],
                "Write Critical Current": [write_critical_current],
                "Write Bias Fraction": [
                    measurement_settings["write_current"] * 1e6 / write_critical_current
                ],
                "Read Current": [measurement_settings["read_current"] * 1e6],
                "Read Critical Current": [read_critical_current],
                "Read Bias Fraction": [
                    measurement_settings["read_current"] * 1e6 / read_critical_current
                ],
                "Bit Error Rate": [data_dict["bit_error_rate"]],
                "Write 0 Read 1": [data_dict["write_0_read_1"]],
                "Write 1 Read 0": [data_dict["write_1_read_0"]],
                "Enable Write Power [uW]": [enable_write_power * 1e6],
                "Enable Read Power [uW]": [enable_read_power * 1e6],
            }
            pd.set_option("display.float_format", "{:.3f}".format)
            param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
            param_df.columns = ["Value"]
            print(param_df)

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    final_dict = {
        **measurement_settings,
        **save_dict,
    }
    return save_dict


def run_sweep_subset(
    b: nTron,
    measurement_settings: dict,
    parameter_x: str,
    parameter_y: str,
    save_traces: bool = False,
    plot_measurement=False,
):
    save_dict = {}
    for idx, x in enumerate(measurement_settings["x"]):
        for y in measurement_settings["y"]:
            measurement_settings[parameter_x] = x
            measurement_settings[parameter_y] = y
            measurement_settings = calculate_voltage(measurement_settings)
            if (
                y > measurement_settings["y_subset"][idx][0]
                and y < measurement_settings["y_subset"][idx][-1]
            ):
                data_dict = run_measurement(
                    b,
                    measurement_settings,
                    save_traces,
                    plot=plot_measurement,
                    ramp_read=False,
                )
                data_dict.update(measurement_settings)
                print(f"bit_error_rate: {data_dict['bit_error_rate']}")
                print(f"write_0_read_1: {data_dict['write_0_read_1']}")
                print(f"write_1_read_0: {data_dict['write_1_read_0']}")
                print("-")
                print(f"write_current [uA]: {data_dict['write_current']*1e6:.2f}")
                print(f"read_current [uA]: {data_dict['read_current']*1e6:.2f}")
                print(
                    f"enable_write_current [uA]: {data_dict['enable_write_current']*1e6:.2f}"
                )
                print(
                    f"enable_read_current [uA]: {data_dict['enable_read_current']*1e6:.2f}"
                )
                print("-")
                print(f"wr_ratio: {data_dict['wr_ratio']}")
                print(f"ewr_ratio: {data_dict['ewr_ratio']}")
                print("-----------------")
            else:
                data_dict = {
                    "trace_chan_in": np.empty(
                        (2, measurement_settings["num_samples_scope"])
                    ),
                    "trace_chan_out": np.empty(
                        (2, measurement_settings["num_samples_scope"])
                    ),
                    "trace_enab": np.empty(
                        (2, measurement_settings["num_samples_scope"])
                    ),
                    "write_0_read_1": np.nan,
                    "write_1_read_0": np.nan,
                    "write_0_read_1_norm": np.nan,
                    "write_1_read_0_norm": np.nan,
                    "read_zero_top": np.empty((1, measurement_settings["num_meas"])),
                    "read_one_top": np.empty((1, measurement_settings["num_meas"])),
                    "channel_voltage": np.nan,
                    "enable_voltage": np.nan,
                    "bit_error_rate": np.nan,
                }

            data_dict.update(measurement_settings)

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    final_dict = {
        **measurement_settings,
        **save_dict,
    }
    return save_dict


def plot_ber_sweep(
    save_dict: dict, measurement_settings: dict, file_path: str, A: str, B: str, C: str
) -> None:
    x = measurement_settings["x"]
    y = measurement_settings["y"]
    if len(x) > 1 and len(y) > 1:
        ax = plot_array(save_dict, C, A, B)
        plt.savefig(file_path + "_0.png")

        plot_array(
            save_dict,
            "write_0_read_1",
            A,
            B,
            cmap=plt.get_cmap("Reds", 100),
            norm=False,
        )
        plt.savefig(file_path + "_1.png")

        plt.show()
        plot_array(
            save_dict,
            "write_1_read_0",
            A,
            B,
            cmap=plt.get_cmap("Blues", 100),
            norm=False,
        )
        plt.savefig(file_path + "_2.png")

        plt.show()

    if len(x) == 1 and len(y) > 1:
        ax = plot_parameter(save_dict, B, C, color="#808080")
        ax = plot_parameter(
            save_dict, B, "write_0_read_1_norm", ax=ax, color=(0.68, 0.12, 0.1)
        )
        ax = plot_parameter(
            save_dict, B, "write_1_read_0_norm", ax=ax, color=(0.16, 0.21, 0.47)
        )
        plt.savefig(file_path + "_3.png")

    if len(y) == 1 and len(x) > 1:
        ax = plot_parameter(save_dict, A, C, color="#808080")
        ax = plot_parameter(
            save_dict, A, "write_0_read_1_norm", ax=ax, color=(0.68, 0.12, 0.1)
        )
        ax = plot_parameter(
            save_dict, A, "write_1_read_0_norm", ax=ax, color=(0.16, 0.21, 0.47)
        )
        plt.savefig(file_path + "_4.png")
