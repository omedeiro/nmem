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
from typing import Dict

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


def extract_switching_parameters(results_dir: Path) -> Dict[str, float]:
    """Extract switching and retrapping currents from I-V characteristic data.

    Returns:
        Dictionary with switching and retrapping parameters
    """

    try:
        import ltspice
    except ImportError:
        print("❌ ltspice package not available for parameter extraction")
        return {}

    simulations_dir = results_dir / "simulations"
    if not simulations_dir.exists():
        print("❌ Simulations directory not found")
        return {}

    # Find .raw files (exclude .op.raw files)
    all_raw_files = [
        f for f in simulations_dir.glob("*.raw") if not f.name.endswith(".op.raw")
    ]
    if not all_raw_files:
        print("❌ No transient .raw files found")
        return {}

    # Use the first simulation file
    raw_file = all_raw_files[0]

    try:
        # Load and parse simulation data
        ltsp = ltspice.Ltspice(str(raw_file))
        ltsp.parse()

        # Get time, voltage and current data
        time = ltsp.get_time()
        voltage = ltsp.get_data("V(out)") * 1e3  # Convert to mV
        current = ltsp.get_data("I(I2)") * 1e6  # Convert to µA

        # Define voltage threshold for switching detection (e.g., 1 mV)
        voltage_threshold = 1.0  # mV

        # Separate the data into ramp up and ramp down phases
        # Find the peak current point to separate phases
        max_current_idx = np.argmax(np.abs(current))

        # Ramp up phase: beginning to peak
        ramp_up_current = current[: max_current_idx + 1]
        ramp_up_voltage = voltage[: max_current_idx + 1]

        # Ramp down phase: peak to end
        ramp_down_current = current[max_current_idx:]
        ramp_down_voltage = voltage[max_current_idx:]

        # Find switching current (during ramp up)
        # Look for where voltage first exceeds threshold during current increase
        switching_current_pos = None
        switching_current_neg = None

        # Positive sweep switching (ramp up)
        pos_mask = ramp_up_current > 0
        if np.any(pos_mask):
            pos_current = ramp_up_current[pos_mask]
            pos_voltage = ramp_up_voltage[pos_mask]
            switched_indices = np.where(pos_voltage > voltage_threshold)[0]
            if len(switched_indices) > 0:
                switching_current_pos = pos_current[switched_indices[0]]

        # Negative sweep switching (ramp up in absolute value)
        neg_mask = ramp_up_current < 0
        if np.any(neg_mask):
            neg_current = np.abs(ramp_up_current[neg_mask])
            neg_voltage = np.abs(ramp_up_voltage[neg_mask])
            switched_indices = np.where(neg_voltage > voltage_threshold)[0]
            if len(switched_indices) > 0:
                switching_current_neg = neg_current[switched_indices[0]]

        # Find retrapping current (during ramp down)
        # Look for where voltage drops below threshold during current decrease
        retrapping_current_pos = None
        retrapping_current_neg = None

        # Positive sweep retrapping (ramp down)
        pos_mask = ramp_down_current > 0
        if np.any(pos_mask):
            pos_current = ramp_down_current[pos_mask]
            pos_voltage = ramp_down_voltage[pos_mask]
            # Look for where voltage drops below threshold (device returns to superconducting)
            retrap_indices = np.where(pos_voltage < voltage_threshold)[0]
            if len(retrap_indices) > 0:
                # Take the current value just before voltage drops below threshold
                if retrap_indices[0] > 0:
                    retrapping_current_pos = pos_current[retrap_indices[0] - 1]
                else:
                    retrapping_current_pos = pos_current[retrap_indices[0]]

        # Negative sweep retrapping (ramp down in absolute value)
        neg_mask = ramp_down_current < 0
        if np.any(neg_mask):
            neg_current = np.abs(ramp_down_current[neg_mask])
            neg_voltage = np.abs(ramp_down_voltage[neg_mask])
            retrap_indices = np.where(neg_voltage < voltage_threshold)[0]
            if len(retrap_indices) > 0:
                if retrap_indices[0] > 0:
                    retrapping_current_neg = neg_current[retrap_indices[0] - 1]
                else:
                    retrapping_current_neg = neg_current[retrap_indices[0]]

        # Calculate averages (should be symmetric)
        switching_currents = [
            x for x in [switching_current_pos, switching_current_neg] if x is not None
        ]
        retrapping_currents = [
            x for x in [retrapping_current_pos, retrapping_current_neg] if x is not None
        ]

        parameters = {}

        if switching_currents:
            parameters["switching_current_avg"] = np.mean(switching_currents)
            parameters["switching_current_pos"] = (
                switching_current_pos if switching_current_pos else np.nan
            )
            parameters["switching_current_neg"] = (
                switching_current_neg if switching_current_neg else np.nan
            )
            parameters["switching_asymmetry"] = (
                (
                    abs(switching_current_pos - switching_current_neg)
                    / np.mean(switching_currents)
                    * 100
                )
                if len(switching_currents) == 2
                else 0
            )

        if retrapping_currents:
            parameters["retrapping_current_avg"] = np.mean(retrapping_currents)
            parameters["retrapping_current_pos"] = (
                retrapping_current_pos if retrapping_current_pos else np.nan
            )
            parameters["retrapping_current_neg"] = (
                retrapping_current_neg if retrapping_current_neg else np.nan
            )
            parameters["retrapping_asymmetry"] = (
                (
                    abs(retrapping_current_pos - retrapping_current_neg)
                    / np.mean(retrapping_currents)
                    * 100
                )
                if len(retrapping_currents) == 2
                else 0
            )

        # Validate that retrapping < switching (as it should be)
        if (
            "switching_current_avg" in parameters
            and "retrapping_current_avg" in parameters
        ):
            # Calculate hysteresis
            parameters["hysteresis_current"] = (
                parameters["switching_current_avg"]
                - parameters["retrapping_current_avg"]
            )
            parameters["hysteresis_percentage"] = (
                parameters["hysteresis_current"]
                / parameters["switching_current_avg"]
                * 100
            )

            if (
                parameters["retrapping_current_avg"]
                > parameters["switching_current_avg"]
            ):
                print(
                    "⚠️  Warning: Retrapping current > switching current - this may indicate analysis error"
                )

        # Print results
        print("\n" + "=" * 50)
        print("🔍 HTRON SWITCHING PARAMETERS")
        print("=" * 50)

        if "switching_current_avg" in parameters:
            print(f"Switching Current (Ramp Up):")
            print(f"  • Average: {parameters['switching_current_avg']:.1f} µA")
            if not np.isnan(parameters["switching_current_pos"]):
                print(
                    f"  • Positive sweep: {parameters['switching_current_pos']:.1f} µA"
                )
            if not np.isnan(parameters["switching_current_neg"]):
                print(
                    f"  • Negative sweep: {parameters['switching_current_neg']:.1f} µA"
                )
            print(f"  • Asymmetry: {parameters['switching_asymmetry']:.1f}%")

        if "retrapping_current_avg" in parameters:
            print(f"\nRetrapping Current (Ramp Down):")
            print(f"  • Average: {parameters['retrapping_current_avg']:.1f} µA")
            if not np.isnan(parameters["retrapping_current_pos"]):
                print(
                    f"  • Positive sweep: {parameters['retrapping_current_pos']:.1f} µA"
                )
            if not np.isnan(parameters["retrapping_current_neg"]):
                print(
                    f"  • Negative sweep: {parameters['retrapping_current_neg']:.1f} µA"
                )
            print(f"  • Asymmetry: {parameters['retrapping_asymmetry']:.1f}%")

        # Print hysteresis information
        if "hysteresis_current" in parameters:
            print(f"\nHysteresis:")
            print(f"  • Current window: {parameters['hysteresis_current']:.1f} µA")
            print(f"  • Percentage: {parameters['hysteresis_percentage']:.1f}%")

        print("=" * 50)

        return parameters

    except Exception as e:
        print(f"❌ Error extracting switching parameters: {e}")
        return {}


def create_iv_characteristic_plot(results_dir: Path, output_path: Path = None) -> None:
    """Create current vs voltage characteristic plot from raw simulation files."""

    try:
        import ltspice
    except ImportError:
        print("❌ ltspice package not available for I-V characteristic plotting")
        return

    simulations_dir = results_dir / "simulations"
    if not simulations_dir.exists():
        print("❌ Simulations directory not found")
        return

    # Find .raw files (exclude .op.raw files which are operating point only)
    all_raw_files = [
        f for f in simulations_dir.glob("*.raw") if not f.name.endswith(".op.raw")
    ]
    if not all_raw_files:
        print("❌ No transient .raw files found in simulations directory")
        return

    # Sort files
    all_raw_files.sort()

    print(
        f"📊 Generating I-V characteristic plot from {len(all_raw_files)} simulations..."
    )

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle("hTron I-V Characteristic", fontsize=16, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_raw_files)))

    for i, raw_file in enumerate(all_raw_files):
        try:
            # Extract parameters from filename
            filename = raw_file.stem

            # Load and parse simulation data
            ltsp = ltspice.Ltspice(str(raw_file))
            ltsp.parse()

            # Get voltage and current data
            try:
                voltage = ltsp.get_data("V(out)") * 1e3  # Convert to mV
                current = ltsp.get_data("I(I2)") * 1e6  # Convert to µA

                # Plot I-V characteristic
                ax.plot(
                    voltage,
                    current,
                    color=colors[i],
                    linewidth=2,
                    label=f"{filename.replace('iv_bias_', '').replace('uA_heater_', 'µA, heater ').replace('uA', 'µA')}",
                    alpha=0.8,
                )

            except Exception as e:
                print(f"⚠️  Could not extract voltage/current from {filename}: {e}")
                continue

        except Exception as e:
            print(f"⚠️  Error processing {raw_file.name}: {e}")
            continue

    # Format plot
    ax.set_xlabel("Voltage (mV)", fontsize=12)
    ax.set_ylabel("Current (µA)", fontsize=12)
    ax.set_title("Current vs Voltage Characteristic", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend if there are multiple traces
    if len(all_raw_files) > 1:
        ax.legend(loc="best")

    # Add zero lines
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"iv_characteristic_{timestamp}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📈 I-V characteristic plot saved to: {output_path}")
    plt.close()

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

    # Find .raw files (exclude .op.raw files which are operating point only)
    all_raw_files = [
        f for f in simulations_dir.glob("*.raw") if not f.name.endswith(".op.raw")
    ]
    if not all_raw_files:
        print("❌ No transient .raw files found in simulations directory")
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

            # Try to get ic current (critical current)
            try:
                ic_current = ltsp.get_data("I(ic)") * 1e6  # Convert to µA
                ax2.plot(
                    time,
                    ic_current,
                    color="#2ca02c",  # Green
                    linewidth=2,
                    label="Critical Current (ic)",
                    alpha=0.8,
                )
            except Exception:
                pass

            # Try to get ir current (retrapping current)
            try:
                ir_current = ltsp.get_data("I(ir)") * 1e6  # Convert to µA
                ax2.plot(
                    time,
                    ir_current,
                    color="#d62728",  # Red
                    linewidth=2,
                    label="Retrapping Current (ir)",
                    alpha=0.8,
                )
            except Exception:
                pass

            # Try to get R3 current
            try:
                r3_current = ltsp.get_data("I(R3)") * 1e6  # Convert to µA
                ax2.plot(
                    time,
                    r3_current,
                    color="#ff1493",  # Deep Pink
                    linewidth=2,
                    label="R3 Current",
                    alpha=0.8,
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
        default=True,
        help="Also plot transient results (default: True)",
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

        # Extract switching parameters
        switching_params = extract_switching_parameters(results_dir)

        # I-V characteristic plot
        iv_char_output = results_dir / f"iv_characteristic_{timestamp}.png"
        create_iv_characteristic_plot(results_dir, iv_char_output)

        # Create transient plots if requested
        if args.show_transients:
            plot_transient_iv_results(results_dir, args.transient_bias_currents)

        print(f"\n🎉 Analysis complete! Results from: {csv_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
