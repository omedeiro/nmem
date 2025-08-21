#!/usr/bin/env python3
"""
hTron IV Sweep Results Plotter

This script creates visualization plots from hTron IV sweep analysis results.
It generates IV curves, switching threshold plots, and heater current effects.

Usage:
    python plot_htron_iv_results.py --latest                    # Plot latest results
    python plot_htron_iv_results.py results.csv               # Plot specific results
    python plot_htron_iv_results.py --latest --show-transients # Include transient plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

# Set non-interactive backend for headless operation
import matplotlib

matplotlib.use("Agg")

# Set plotting style
plt.style.use("default")


def find_latest_results(results_dir: Path) -> Path:
    """Find the most recent IV sweep results directory."""
    sweep_dirs = list(results_dir.glob("htron_iv_sweep_*"))
    if not sweep_dirs:
        raise FileNotFoundError("No IV sweep results found in results directory")

    # Sort by directory name (which includes timestamp)
    latest_dir = sorted(sweep_dirs)[-1]
    csv_file = latest_dir / "htron_iv_sweep_results.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"No results CSV found in {latest_dir}")

    return csv_file


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and validate results CSV."""
    try:
        df = pd.read_csv(csv_path)
        required_cols = [
            "Bias_Current_uA",
            "Heater_Current_uA",
            "Max_Voltage_mV",
            "Success",
        ]

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        print(f"📊 Loaded {len(df)} data points from {csv_path}")
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load results: {e}")


def create_iv_curves_plot(df: pd.DataFrame, output_path: Path = None) -> None:
    """Create IV curves for different heater currents."""

    # Filter successful simulations
    df_success = df[df["Success"] == True].copy()

    if len(df_success) == 0:
        print("⚠️  Warning: No successful simulations found. Skipping IV curves plot.")
        return None

    # Get unique heater currents
    heater_currents = sorted(df_success["Heater_Current_uA"].unique())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Color map for different heater currents
    colors = plt.cm.viridis(np.linspace(0, 1, len(heater_currents)))

    # Plot 1: IV curves (bias current vs max voltage)
    for i, heater_i in enumerate(heater_currents):
        df_heater = df_success[df_success["Heater_Current_uA"] == heater_i]

        if len(df_heater) > 0:
            ax1.plot(
                df_heater["Bias_Current_uA"],
                df_heater["Max_Voltage_mV"],
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=6,
                label=f"Heater: {heater_i:.0f} µA",
            )

    ax1.set_xlabel("Bias Current (µA)", fontsize=12)
    ax1.set_ylabel("Max Voltage (mV)", fontsize=12)
    ax1.set_title("hTron IV Characteristics", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Current transfer (bias current vs max measured current)
    for i, heater_i in enumerate(heater_currents):
        df_heater = df_success[df_success["Heater_Current_uA"] == heater_i]

        if len(df_heater) > 0 and "Max_Current_uA" in df_heater.columns:
            ax2.plot(
                df_heater["Bias_Current_uA"],
                df_heater["Max_Current_uA"],
                "s-",
                color=colors[i],
                linewidth=2,
                markersize=6,
                label=f"Heater: {heater_i:.0f} µA",
            )

    ax2.set_xlabel("Bias Current (µA)", fontsize=12)
    ax2.set_ylabel("Max Measured Current (µA)", fontsize=12)
    ax2.set_title("Current Transfer Characteristics", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = csv_path.parent / f"iv_curves_{timestamp}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📈 IV curves plot saved to: {output_path}")

    return fig


def create_switching_threshold_plot(df: pd.DataFrame, output_path: Path = None) -> None:
    """Create plot showing switching threshold vs heater current."""

    # Filter successful simulations
    df_success = df[df["Success"] == True].copy()

    if len(df_success) == 0:
        print("⚠️  Warning: No successful simulations found. Skipping threshold plot.")
        return None

    # Define switching threshold (e.g., where voltage > 1 mV)
    threshold_voltage = 1.0  # mV

    switching_thresholds = []
    heater_currents = sorted(df_success["Heater_Current_uA"].unique())

    for heater_i in heater_currents:
        df_heater = df_success[df_success["Heater_Current_uA"] == heater_i]

        # Find switching threshold (first bias current where voltage exceeds threshold)
        switched = df_heater[df_heater["Max_Voltage_mV"] > threshold_voltage]

        if len(switched) > 0:
            switching_threshold = switched["Bias_Current_uA"].min()
            switching_thresholds.append(switching_threshold)
        else:
            switching_thresholds.append(np.nan)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Remove NaN values for plotting
    valid_data = [
        (h, s) for h, s in zip(heater_currents, switching_thresholds) if not np.isnan(s)
    ]

    if valid_data:
        heater_vals, threshold_vals = zip(*valid_data)

        ax.plot(
            heater_vals,
            threshold_vals,
            "o-",
            linewidth=3,
            markersize=8,
            color="tab:red",
        )

        # Add trend line if we have enough points
        if len(threshold_vals) >= 2:
            z = np.polyfit(heater_vals, threshold_vals, 1)
            p = np.poly1d(z)
            ax.plot(
                heater_vals,
                p(heater_vals),
                "--",
                alpha=0.7,
                color="tab:blue",
                label=f"Linear fit: slope = {z[0]:.2f} µA/µA",
            )
            ax.legend()

    ax.set_xlabel("Heater Current (µA)", fontsize=12)
    ax.set_ylabel("Switching Threshold (µA)", fontsize=12)
    ax.set_title(
        f"hTron Switching Threshold vs Heater Current\n(Threshold: {threshold_voltage} mV)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    if valid_data:
        min_threshold = min(threshold_vals)
        max_threshold = max(threshold_vals)
        threshold_range = max_threshold - min_threshold

        textstr = f"""Data Points: {len(threshold_vals)}
Min Threshold: {min_threshold:.1f} µA
Max Threshold: {max_threshold:.1f} µA
Range: {threshold_range:.1f} µA"""

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    plt.tight_layout()

    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = csv_path.parent / f"switching_threshold_{timestamp}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📈 Switching threshold plot saved to: {output_path}")

    return fig


def plot_transient_iv_results(results_dir: Path, bias_currents: list = None) -> None:
    """Plot transient simulation results from .raw files for IV sweeps."""

    try:
        import ltspice
    except ImportError:
        print("❌ ltspice package not available for transient plotting")
        return

    simulations_dir = results_dir / "simulations"
    if not simulations_dir.exists():
        print("❌ Simulations directory not found")
        return

    # Find .raw files
    all_raw_files = list(simulations_dir.glob("*.raw"))
    if not all_raw_files:
        print("❌ No .raw files found in simulations directory")
        return

    # Filter by specific bias currents if provided
    if bias_currents:
        filtered_files = []
        for bc in bias_currents:
            pattern = f"*bias_{bc:.0f}uA*.raw"
            matching = list(simulations_dir.glob(pattern))
            filtered_files.extend(matching)
        all_raw_files = filtered_files

    if not all_raw_files:
        print("❌ No matching .raw files found")
        return

    # Sort files
    all_raw_files.sort()

    print(f"📊 Generating transient plots for {len(all_raw_files)} simulations...")

    # Standard colors
    colors = {
        "voltage": "#9467bd",  # Purple
        "current": "#1f77b4",  # Blue
        "heater": "#ff7f0e",  # Orange
    }

    plots_generated = 0

    for raw_file in all_raw_files:
        try:
            # Extract parameters from filename
            filename = raw_file.stem
            # Expected format: iv_bias_XXXuA_heater_YYYuA

            # Load and parse simulation data
            ltsp = ltspice.Ltspice(str(raw_file))
            ltsp.parse()

            time = ltsp.get_time() * 1e6  # Convert to microseconds

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(
                f"IV Transient Analysis - {filename}", fontsize=16, fontweight="bold"
            )

            # =============================================================
            # TOP SUBPLOT: Voltages
            # =============================================================
            voltage_plotted = False

            # Try to get output voltage
            voltage_signals = ["V(out)", "V(output)", "V(n001)"]
            for signal_name in voltage_signals:
                try:
                    voltage = ltsp.get_data(signal_name) * 1e3  # Convert to mV
                    ax1.plot(
                        time,
                        voltage,
                        color=colors["voltage"],
                        linewidth=2,
                        label="Output Voltage",
                    )
                    voltage_plotted = True
                    break
                except Exception:
                    continue

            # Try to get heater voltage if available
            try:
                heater_voltage = ltsp.get_data("V(NC_01)") * 1e3  # Convert to mV
                ax1.plot(
                    time,
                    heater_voltage,
                    color=colors["heater"],
                    linewidth=2,
                    label="Heater Voltage",
                    alpha=0.7,
                )
            except Exception:
                pass

            # Voltage subplot formatting
            ax1.set_ylabel("Voltage (mV)", fontsize=12)
            ax1.set_title("Voltage vs Time", fontsize=14)
            ax1.grid(True, alpha=0.3)
            if voltage_plotted:
                ax1.legend(loc="upper right")
                ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

            # =============================================================
            # BOTTOM SUBPLOT: Currents
            # =============================================================
            current_plotted = False

            # Try to get bias current
            current_signals = ["I(I2)", "I(X§U1:Lc)"]
            for signal_name in current_signals:
                try:
                    current = ltsp.get_data(signal_name) * 1e6  # Convert to µA
                    ax2.plot(
                        time,
                        current,
                        color=colors["current"],
                        linewidth=2,
                        label="Bias Current",
                    )
                    current_plotted = True
                    break
                except Exception:
                    continue

            # Try to get heater current
            try:
                heater_current = ltsp.get_data("I(I1)") * 1e6  # Convert to µA
                ax2.plot(
                    time,
                    heater_current,
                    color=colors["heater"],
                    linewidth=2,
                    label="Heater Current",
                    alpha=0.7,
                )
            except Exception:
                pass

            # Current subplot formatting
            ax2.set_xlabel("Time (µs)", fontsize=12)
            ax2.set_ylabel("Current (µA)", fontsize=12)
            ax2.set_title("Current vs Time", fontsize=14)
            ax2.grid(True, alpha=0.3)
            if current_plotted:
                ax2.legend(loc="upper right")
                ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

            plt.tight_layout()

            # Save individual plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"iv_transient_{filename}_{timestamp}.png"
            output_path = results_dir / "plots" / output_filename

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close figure to free memory

            print(f"📈 Generated: {output_filename}")
            plots_generated += 1

        except Exception as e:
            print(f"⚠️  Error plotting {raw_file.name}: {e}")
            continue

    print(f"🎉 Generated {plots_generated} transient plots in {results_dir / 'plots'}")


def print_results_summary(df: pd.DataFrame) -> None:
    """Print a summary of the IV sweep results."""

    print("\n" + "=" * 50)
    print("📊 HTRON IV SWEEP RESULTS SUMMARY")
    print("=" * 50)

    # Filter successful simulations
    df_success = df[df["Success"] == True]

    print(f"Total simulations: {len(df)} ")
    print(f"Successful simulations: {len(df_success)}")
    print(f"Failed simulations: {len(df) - len(df_success)}")

    if len(df_success) > 0:
        print(
            f"Bias current range: {df['Bias_Current_uA'].min():.1f} - {df['Bias_Current_uA'].max():.1f} µA"
        )
        print(
            f"Heater current range: {df['Heater_Current_uA'].min():.1f} - {df['Heater_Current_uA'].max():.1f} µA"
        )

        print(
            f"Max voltage range: {df_success['Max_Voltage_mV'].min():.2f} - {df_success['Max_Voltage_mV'].max():.2f} mV"
        )

        if "Max_Current_uA" in df_success.columns:
            print(
                f"Max current range: {df_success['Max_Current_uA'].min():.2f} - {df_success['Max_Current_uA'].max():.2f} µA"
            )

    print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot hTron IV sweep analysis results")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("csv_file", nargs="?", help="Path to results CSV file")
    group.add_argument(
        "--latest", "-l", action="store_true", help="Use most recent results"
    )

    parser.add_argument("--output", "-o", help="Output plot filename")
    parser.add_argument(
        "--show-transients",
        "-t",
        action="store_true",
        help="Also plot transient results",
    )
    parser.add_argument(
        "--transient-bias-currents",
        nargs="*",
        type=float,
        help="Specific bias currents to show transients for (e.g., 100.0 200.0)",
    )

    args = parser.parse_args()

    # Determine input file
    if args.latest:
        results_dir = Path("../results")
        csv_path = find_latest_results(results_dir)
        results_dir = csv_path.parent
    else:
        csv_path = Path(args.csv_file)
        results_dir = csv_path.parent

    if not csv_path.exists():
        print(f"❌ Error: Results file not found: {csv_path}")
        return

    # Load and analyze data
    try:
        df = load_results(csv_path)
        print_results_summary(df)

        # Create plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # IV curves plot
        iv_output = (
            results_dir / f"iv_curves_{timestamp}.png"
            if not args.output
            else Path(args.output)
        )
        create_iv_curves_plot(df, iv_output)

        # Switching threshold plot
        threshold_output = results_dir / f"switching_threshold_{timestamp}.png"
        create_switching_threshold_plot(df, threshold_output)

        # Create transient plots if requested
        if args.show_transients:
            plot_transient_iv_results(results_dir, args.transient_bias_currents)

        print(f"\n🎉 Analysis complete! Results from: {csv_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
