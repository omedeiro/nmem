import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nmem.analysis.histogram_utils import plot_general_histogram


def plot_alignment_histogram(
    diff_list,
    binwidth=1,
    save_fig=False,
    output_path="alignment_histogram.pdf",
    ax=None,
) -> tuple[plt.Figure, Axes]:
    """
    Plots a histogram of alignment differences.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    bins = range(
        int(np.floor(min(diff_list))),
        int(np.ceil(max(diff_list))) + binwidth,
        binwidth,
    )
    n, bins, patches = plot_general_histogram(
        diff_list,
        bins=bins,
        color="#0504aa",
        alpha=0.5,
        edgecolor="black",
        xlabel="alignment difference [nm]",
        ylabel="count",
        legend=False,
        ax=ax,
    )
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_alignment_offset_hist(
    dx_nm, dy_nm, save_fig=False, output_path="alignment_offsets_histogram.pdf", ax=None
) -> tuple[plt.Figure, Axes]:
    """
    Plots histograms of alignment offsets for ΔX and ΔY.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    plot_general_histogram(
        dx_nm,
        bins=20,
        color="#1f77b4",
        alpha=0.7,
        edgecolor="black",
        label="ΔX [nm]",
        xlabel="Alignment Offset [nm]",
        ylabel="Count",
        legend=True,
        grid=True,
        ax=ax,
    )
    plot_general_histogram(
        dy_nm,
        bins=20,
        color="#ff7f0e",
        alpha=0.7,
        edgecolor="black",
        label="ΔY [nm]",
        legend=True,
        grid=True,
        ax=ax,
    )
    ax.set_title("Histogram of Alignment Offsets")
    if save_fig:
        fig.savefig(output_path, dpi=300)
    return fig, ax


def plot_alignment_stats(
    df_z,
    df_rot_valid,
    dx_nm,
    dy_nm,
    z_mean,
    z_std,
    r_mean,
    r_std,
    save=False,
    output_path="alignment_analysis.pdf",
    axs=None,
) -> tuple[plt.Figure, list[Axes]]:
    """
    Plots histograms and KDE for alignment statistics (z height, rotation, and alignment offsets).
    """
    import seaborn as sns

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    else:
        fig = axs[0].figure

    # Z height histogram
    plot_general_histogram(
        df_z["z_height_mm"],
        bins=20,
        color="#1f77b4",
        edgecolor="black",
        xlabel="Z Height [mm]",
        ylabel="Count",
        legend=False,
        ax=axs[0],
    )
    axs[0].text(
        0.97,
        0.97,
        f"$\\mu$ = {z_mean:.4f} mm\n$\\sigma$ = {z_std:.4f} mm",
        transform=axs[0].transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
    )
    # Rotation histogram
    plot_general_histogram(
        df_rot_valid["rotation_mrad"],
        bins=20,
        color="#1f77b4",
        edgecolor="black",
        xlabel="Rotation [mrad]",
        ylabel="Count",
        legend=False,
        ax=axs[1],
    )
    axs[1].text(
        0.97,
        0.97,
        f"$\\mu$ = {r_mean:.2f} mrad\n$\\sigma$ = {r_std:.2f} mrad",
        transform=axs[1].transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
    )
    # Alignment offsets KDE and scatter
    ax = axs[2]
    sns.kdeplot(
        x=dx_nm,
        y=dy_nm,
        fill=True,
        cmap="crest",
        bw_adjust=0.7,
        levels=10,
        thresh=0.05,
        ax=ax,
    )
    ax.scatter(
        dx_nm,
        dy_nm,
        color="#333333",
        s=15,
        marker="o",
        label="Alignment Marks",
        alpha=0.8,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ΔX [nm]")
    ax.set_ylabel("ΔY [nm]")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=300)
    return fig, axs
