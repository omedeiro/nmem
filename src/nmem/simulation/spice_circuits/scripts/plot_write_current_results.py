#!/usr/bin/env python3
"""
Write Current Sweep Results Plotter

This script creates simple visualization plots from write current sweep analysis results.
It shows write current vs persistent current and displays transient simulation results by default.

Usage:
    python plot_write_current_results.py --latest                    # Plot sweep results with transients
    python plot_write_current_results.py --latest --no-transients    # Plot sweep results only
    python plot_write_current_results.py results.csv
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
    """Find the most recent sweep results directory."""
    sweep_dirs = list(results_dir.glob("write_current_sweep_*"))
    if not sweep_dirs:
        raise FileNotFoundError("No sweep results found in results directory")

    # Sort by directory name (which includes timestamp)
    latest_dir = sorted(sweep_dirs)[-1]
    csv_file = latest_dir / "write_current_sweep_results.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"No results CSV found in {latest_dir}")

    return csv_file


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and validate results CSV."""
    try:
        df = pd.read_csv(csv_path)
        required_cols = [
            "Write_Current_uA",
            "Persistent_Current_uA",
            "Efficiency_Percent",
        ]

        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        print(f"📊 Loaded {len(df)} data points from {csv_path}")
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load results: {e}")


def create_simple_plot(df: pd.DataFrame, output_path: Path = None) -> None:
    """Create a simple write current vs persistent current plot."""

    # Filter out NaN values from the data
    df_clean = df.dropna(subset=["Persistent_Current_uA"])

    if len(df_clean) == 0:
        print(
            "⚠️  Warning: No valid persistent current data found. Skipping persistent current plot."
        )
        return None

    if len(df_clean) < len(df):
        print(
            f"⚠️  Warning: Filtered out {len(df) - len(df_clean)} NaN values from persistent current data."
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot data points with both magnitude and signed values
    ax.plot(
        df_clean["Write_Current_uA"],
        df_clean["Persistent_Current_uA"],
        "o-",
        linewidth=3,
        markersize=10,
        color="tab:blue",
        label="Persistent Current",
    )

    # Also plot absolute values for comparison
    ax.plot(
        df_clean["Write_Current_uA"],
        np.abs(df_clean["Persistent_Current_uA"]),
        "s--",
        linewidth=2,
        markersize=8,
        color="tab:orange",
        alpha=0.7,
        label="|Persistent Current|",
    )

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add trend line for signed values (only if we have enough data points)
    if len(df_clean) >= 2:
        try:
            z = np.polyfit(
                df_clean["Write_Current_uA"], df_clean["Persistent_Current_uA"], 1
            )
            p = np.poly1d(z)
            ax.plot(
                df_clean["Write_Current_uA"],
                p(df_clean["Write_Current_uA"]),
                "--",
                alpha=0.5,
                color="tab:red",
                label=f"Linear fit: {z[0]:.3f}x + {z[1]:.1f}",
            )
        except np.RankWarning:
            print("⚠️  Warning: Could not fit linear trend line to data")

    ax.set_xlabel("Write Current (µA)", fontsize=14)
    ax.set_ylabel("Persistent Current (µA)", fontsize=14)
    ax.set_title(
        "Memory Cell: Write Current vs Persistent Current",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add text box with key statistics
    max_write = df_clean["Write_Current_uA"].max()
    max_persistent = df_clean["Persistent_Current_uA"].iloc[-1]
    efficiency = (abs(max_persistent) / max_write) * 100 if max_write != 0 else 0

    # Calculate slope safely
    if len(df_clean) >= 2:
        try:
            z = np.polyfit(
                df_clean["Write_Current_uA"], df_clean["Persistent_Current_uA"], 1
            )
            slope_text = f"Slope: {z[0]:.3f} µA/µA"
        except:
            slope_text = "Slope: N/A"
    else:
        slope_text = "Slope: N/A (insufficient data)"

    textstr = f"""Data Points: {len(df_clean)} (of {len(df)})
Write Range: {df_clean['Write_Current_uA'].min():.0f}-{df_clean['Write_Current_uA'].max():.0f} µA
Max |Persistent|: {abs(df_clean['Persistent_Current_uA']).max():.1f} µA
{slope_text}
Efficiency: {efficiency:.1f}%"""

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📈 Plot saved to: {output_path}")

    return fig


def plot_transient_results(results_dir: Path, write_currents: list = None) -> None:
    """Plot transient simulation results from .raw files in standard format.

    Generates individual PNG files for each write current value with:
    - Top subplot: Left and Right branch currents
    - Bottom subplot: Output voltage
    """

    try:
        import ltspice
    except ImportError:
        print("❌ ltspice package not available for transient plotting")
        return

    simulations_dir = results_dir / "simulations"
    if not simulations_dir.exists():
        print("❌ Simulations directory not found")
        return

    # Find .raw files (exclude .op.raw files which are operating point analysis)
    all_raw_files = list(simulations_dir.glob("*.raw"))
    raw_files = [f for f in all_raw_files if not f.name.endswith(".op.raw")]
    if not raw_files:
        print("❌ No transient .raw files found in simulations directory")
        return

    # Limit to specific write currents if provided
    if write_currents:
        filtered_files = []
        for wc in write_currents:
            pattern = f"*{wc:04.1f}uA.raw"
            matching = [
                f
                for f in simulations_dir.glob(pattern)
                if not f.name.endswith(".op.raw")
            ]
            filtered_files.extend(matching)
        raw_files = filtered_files

    if not raw_files:
        print("❌ No matching .raw files found")
        return

    # Sort files by write current
    raw_files.sort()

    print(
        f"📊 Generating individual transient plots for {len(raw_files)} simulations..."
    )

    # Standard colors matching unified plotter
    colors = {
        "left": "#1f77b4",  # Blue
        "right": "#ff7f0e",  # Orange
        "voltage": "#9467bd",  # Purple
    }

    plots_generated = 0

    for raw_file in raw_files:
        try:
            # Extract write current from filename
            filename = raw_file.stem
            write_current_str = filename.split("_")[-1].replace("uA", "")

            try:
                write_current_val = float(write_current_str)
            except ValueError:
                print(f"⚠️  Could not parse write current from {filename}")
                continue

            # Load and parse simulation data
            ltsp = ltspice.Ltspice(str(raw_file))
            ltsp.parse()

            time = ltsp.get_time() * 1e9  # Convert to nanoseconds

            # Create figure with 5 subplots (current, voltage, critical currents, retrapping currents, temperature)
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))
            fig.suptitle(
                f"Transient Analysis - Write Current: {write_current_str} μA",
                fontsize=16,
                fontweight="bold",
            )

            # =============================================================
            # TOP SUBPLOT: CURRENT (Left and Right branch currents)
            # =============================================================
            current_plotted = False

            # Try to get left branch current
            try:
                left_current = ltsp.get_data("Ix(hl:drain)") * 1e6  # Convert to µA
                ax1.plot(
                    time,
                    left_current,
                    color=colors["left"],
                    linewidth=2,
                    label="Left Current",
                )
                current_plotted = True
            except Exception:
                # Try alternative left current signal
                try:
                    left_current = (
                        ltsp.get_data("Ix(HL:drain)") * 1e6
                    )  # Uppercase version
                    ax1.plot(
                        time,
                        left_current,
                        color=colors["left"],
                        linewidth=2,
                        label="Left Current",
                    )
                    current_plotted = True
                except Exception:
                    print(f"⚠️  Left current not found in {filename}")

            # Try to get right branch current
            try:
                right_current = ltsp.get_data("Ix(hr:drain)") * 1e6  # Convert to µA
                ax1.plot(
                    time,
                    right_current,
                    color=colors["right"],
                    linewidth=2,
                    label="Right Current",
                )
                current_plotted = True
            except Exception:
                # Try alternative right current signal
                try:
                    right_current = (
                        ltsp.get_data("Ix(HR:drain)") * 1e6
                    )  # Uppercase version
                    ax1.plot(
                        time,
                        right_current,
                        color=colors["right"],
                        linewidth=2,
                        label="Right Current",
                    )
                    current_plotted = True
                except Exception:
                    print(f"⚠️  Right current not found in {filename}")

            # Current subplot formatting
            ax1.set_ylabel("Current (μA)", fontsize=12)
            ax1.set_title("Branch Currents", fontsize=14)
            ax1.grid(True, alpha=0.3)
            if current_plotted:
                ax1.legend(loc="upper right")
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No current data available",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )

            # =============================================================
            # MIDDLE SUBPLOT: VOLTAGE (Output voltage)
            # =============================================================
            voltage_plotted = False

            # Try different voltage signal names (prioritize V(out) since it's available)
            voltage_signals = ["V(out)", "V(output)", "V(vout)", "V(n001)", "V(n002)"]

            for signal_name in voltage_signals:
                try:
                    voltage = ltsp.get_data(signal_name) * 1e3  # Convert to mV
                    ax2.plot(
                        time,
                        voltage,
                        color=colors["voltage"],
                        linewidth=2,
                        label="Output Voltage",
                    )
                    voltage_plotted = True
                    break  # Found voltage signal, stop trying
                except Exception:
                    continue

            # If standard names didn't work, try to find any voltage signal
            if not voltage_plotted:
                try:
                    # Get all available signals and look for voltage-like names
                    available_signals = ltsp.get_variable_names()
                    for signal in available_signals:
                        if signal.startswith("V(") and "time" not in signal.lower():
                            try:
                                voltage = ltsp.get_data(signal) * 1e3  # Convert to mV
                                ax2.plot(
                                    time,
                                    voltage,
                                    color=colors["voltage"],
                                    linewidth=2,
                                    label=f"Voltage ({signal})",
                                )
                                voltage_plotted = True
                                break
                            except Exception:
                                continue
                except Exception:
                    pass

            # Voltage subplot formatting
            ax2.set_ylabel("Voltage (mV)", fontsize=12)
            ax2.set_title("Output Voltage", fontsize=14)
            ax2.grid(True, alpha=0.3)

            if voltage_plotted:
                ax2.legend(loc="upper right")
                # Add zero reference line
                ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No voltage data available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )

            # =============================================================
            # THIRD SUBPLOT: CRITICAL CURRENTS (Left and Right)
            # =============================================================
            critical_plotted = False

            # Add colors for critical currents
            colors["critical_left"] = "#17becf"  # Cyan
            colors["critical_right"] = "#ff7f0e"  # Orange

            # Try to get left critical current
            try:
                left_critical = ltsp.get_data("I(ichl)") * 1e6  # Convert to µA
                ax3.plot(
                    time,
                    left_critical,
                    color=colors["critical_left"],
                    linewidth=2,
                    label="Left Critical Current",
                )
                critical_plotted = True
            except Exception:
                try:
                    # Try alternative naming
                    left_critical = ltsp.get_data("I(Ichl)") * 1e6  # Convert to µA
                    ax3.plot(
                        time,
                        left_critical,
                        color=colors["critical_left"],
                        linewidth=2,
                        label="Left Critical Current",
                    )
                    critical_plotted = True
                except Exception:
                    print(
                        f"⚠️  Left critical current not found in {filename}: I(ichl) or I(Ichl)"
                    )

            # Try to get right critical current
            try:
                right_critical = ltsp.get_data("I(ichr)") * 1e6  # Convert to µA
                ax3.plot(
                    time,
                    right_critical,
                    color=colors["critical_right"],
                    linewidth=2,
                    label="Right Critical Current",
                )
                critical_plotted = True
            except Exception:
                try:
                    # Try alternative naming
                    right_critical = ltsp.get_data("I(Ichr)") * 1e6  # Convert to µA
                    ax3.plot(
                        time,
                        right_critical,
                        color=colors["critical_right"],
                        linewidth=2,
                        label="Right Critical Current",
                    )
                    critical_plotted = True
                except Exception:
                    print(
                        f"⚠️  Right critical current not found in {filename}: I(ichr) or I(Ichr)"
                    )

            # Critical currents subplot formatting
            ax3.set_ylabel("Current (μA)", fontsize=12)
            ax3.set_title("Critical Currents", fontsize=14)
            ax3.grid(True, alpha=0.3)

            if critical_plotted:
                ax3.legend(loc="upper right")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No critical current data available",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )

            # =============================================================
            # FOURTH SUBPLOT: RETRAPPING CURRENTS (Left and Right)
            # =============================================================
            retrapping_plotted = False

            # Add colors for retrapping currents
            colors["retrapping_left"] = "#2ca02c"  # Green
            colors["retrapping_right"] = "#d62728"  # Red

            # Try to get left retrapping current
            try:
                left_retrapping = ltsp.get_data("I(irhl)") * 1e6  # Convert to µA
                ax4.plot(
                    time,
                    left_retrapping,
                    color=colors["retrapping_left"],
                    linewidth=2,
                    label="Left Retrapping Current",
                )
                retrapping_plotted = True
            except Exception:
                print(f"⚠️  Left retrapping current not found in {filename}: I(irhl)")

            # Try to get right retrapping current
            try:
                right_retrapping = ltsp.get_data("I(irhr)") * 1e6  # Convert to µA
                ax4.plot(
                    time,
                    right_retrapping,
                    color=colors["retrapping_right"],
                    linewidth=2,
                    label="Right Retrapping Current",
                )
                retrapping_plotted = True
            except Exception:
                print(f"⚠️  Right retrapping current not found in {filename}: I(irhr)")

            # Retrapping currents subplot formatting
            ax4.set_ylabel("Current (μA)", fontsize=12)
            ax4.set_title("Retrapping Currents", fontsize=14)
            ax4.grid(True, alpha=0.3)

            if retrapping_plotted:
                ax4.legend(loc="upper right")
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No retrapping current data available",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )

            # =============================================================
            # FIFTH SUBPLOT: TEMPERATURE (Left and Right hTron Temperatures)
            # =============================================================
            temperature_plotted = False

            # Add colors for temperature traces
            colors["temp_left"] = "#8c564b"  # Brown
            colors["temp_right"] = "#e377c2"  # Pink

            # Try to get temperature measurements
            # Based on the circuit, temperature nodes are tempL and tempR
            try:
                left_temp = ltsp.get_data("V(tempL)")  # Temperature in Kelvin
                ax5.plot(
                    time,
                    left_temp,
                    color=colors["temp_left"],
                    linewidth=2,
                    label="Left hTron Temperature",
                )
                temperature_plotted = True
            except Exception:
                print(f"⚠️  Left temperature V(tempL) not found in {filename}")

            try:
                right_temp = ltsp.get_data("V(tempR)")  # Temperature in Kelvin
                ax5.plot(
                    time,
                    right_temp,
                    color=colors["temp_right"],
                    linewidth=2,
                    label="Right hTron Temperature",
                )
                temperature_plotted = True
            except Exception:
                print(f"⚠️  Right temperature V(tempR) not found in {filename}")

            # If primary signals not found, try case variations
            if not temperature_plotted:
                temperature_signals = ["V(templ)", "V(tempr)", "V(TEMPL)", "V(TEMPR)"]
                for i, signal_name in enumerate(temperature_signals):
                    try:
                        temperature = ltsp.get_data(
                            signal_name
                        )  # Temperature in Kelvin
                        label = (
                            "Left hTron Temperature"
                            if i % 2 == 0
                            else "Right hTron Temperature"
                        )
                        color = (
                            colors["temp_left"] if i % 2 == 0 else colors["temp_right"]
                        )
                        ax5.plot(
                            time,
                            temperature,
                            color=color,
                            linewidth=2,
                            label=label,
                        )
                        temperature_plotted = True
                    except Exception:
                        continue

            # If still no temperature found, try to find any temperature-like signal
            if not temperature_plotted:
                try:
                    available_signals = ltsp.get_variable_names()
                    for signal in available_signals:
                        if "temp" in signal.lower() or "tch" in signal.lower():
                            try:
                                temp_data = ltsp.get_data(signal)
                                ax5.plot(
                                    time,
                                    temp_data,
                                    color=colors["temp_left"],
                                    linewidth=2,
                                    label=f"Temperature ({signal})",
                                )
                                temperature_plotted = True
                                break
                            except Exception:
                                continue
                except Exception:
                    pass

            # Temperature subplot formatting
            ax5.set_xlabel("Time (ns)", fontsize=12)  # Only bottom subplot has x-label
            ax5.set_ylabel("Temperature (K)", fontsize=12)
            ax5.set_title("hTron Channel Temperature", fontsize=14)
            ax5.grid(True, alpha=0.3)

            if temperature_plotted:
                ax5.legend(loc="upper right")
            else:
                ax5.text(
                    0.5,
                    0.5,
                    "No temperature data available",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                )

                # Debug: print available signals to help identify temperature signals
                try:
                    available_signals = ltsp.get_variable_names()
                    temp_related = [
                        s
                        for s in available_signals
                        if any(term in s.lower() for term in ["temp", "tch", "meas"])
                    ]
                    if temp_related:
                        print(
                            f"⚠️  Temperature-related signals found in {filename}: {temp_related}"
                        )
                    else:
                        print(
                            f"⚠️  No temperature signals found in {filename}. Available signals: {available_signals[:10]}..."
                        )
                except Exception:
                    print(f"⚠️  Could not list available signals in {filename}")

            plt.tight_layout()

            # Save individual plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = (
                f"standard_transient_{write_current_str}uA_{timestamp}.png"
            )
            output_path = results_dir / output_filename

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()  # Close figure to free memory

            print(f"📈 Generated: {output_filename}")
            plots_generated += 1

        except Exception as e:
            print(f"⚠️  Error plotting {raw_file.name}: {e}")
            continue

    print(f"🎉 Generated {plots_generated} individual transient plots in {results_dir}")


def print_simple_summary(df: pd.DataFrame) -> None:
    """Print a simple text summary of the analysis results."""

    print("\n" + "=" * 50)
    print("📊 WRITE CURRENT SWEEP RESULTS")
    print("=" * 50)

    # Filter out NaN values for summary calculations
    df_clean = df.dropna(subset=["Persistent_Current_uA"])

    print(
        f"Data points: {len(df)} total, {len(df_clean)} valid persistent current values"
    )
    print(
        f"Write current range: {df['Write_Current_uA'].min():.1f} - {df['Write_Current_uA'].max():.1f} µA"
    )

    if len(df_clean) > 0:
        print(
            f"Persistent current range: {df_clean['Persistent_Current_uA'].min():.2f} - {df_clean['Persistent_Current_uA'].max():.2f} µA"
        )

        # Linear fit (only if we have enough valid data points)
        if len(df_clean) >= 2:
            try:
                coeffs = np.polyfit(
                    df_clean["Write_Current_uA"], df_clean["Persistent_Current_uA"], 1
                )
                print(f"Linear slope: {coeffs[0]:.3f} µA persistent / µA write")
                print(f"Memory efficiency: {abs(coeffs[0]*100):.1f}%")
            except:
                print("Linear slope: N/A (could not fit)")
                print("Memory efficiency: N/A")
        else:
            print("Linear slope: N/A (insufficient data)")
            print("Memory efficiency: N/A")

        # Best efficiency point (only for clean data)
        if len(df_clean) > 0 and "Efficiency_Percent" in df_clean.columns:
            valid_efficiency = df_clean.dropna(subset=["Efficiency_Percent"])
            if len(valid_efficiency) > 0:
                max_eff_idx = np.abs(valid_efficiency["Efficiency_Percent"]).idxmax()
                optimal_write = valid_efficiency.loc[max_eff_idx, "Write_Current_uA"]
                max_efficiency = np.abs(
                    valid_efficiency.loc[max_eff_idx, "Efficiency_Percent"]
                )
                print(
                    f"Maximum efficiency: {max_efficiency:.1f}% at {optimal_write:.1f}µA"
                )
            else:
                print("Maximum efficiency: N/A (no valid efficiency data)")
    else:
        print("Persistent current range: N/A (no valid data)")
        print("Linear slope: N/A (no valid data)")
        print("Memory efficiency: N/A (no valid data)")
        print("Maximum efficiency: N/A (no valid data)")

    print("=" * 50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plot write current sweep analysis results"
    )

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
        help="Also plot transient results (enabled by default)",
    )
    parser.add_argument(
        "--no-transients",
        action="store_true",
        help="Disable transient plots",
    )
    parser.add_argument(
        "--transient-currents",
        nargs="*",
        type=float,
        help="Specific write currents to show transients for (e.g., 60.0 80.0)",
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
        print_simple_summary(df)

        # Create main plot (skip if no valid data)
        if args.output:
            output_path = Path(args.output)
        else:
            # Save to results directory by default
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"write_current_vs_persistent_{timestamp}.png"

        persistent_plot_created = create_simple_plot(df, output_path)
        if persistent_plot_created is None:
            print("⚠️  Skipping persistent current plot due to NaN values")

        # Create transient plots by default (unless disabled)
        if args.show_transients and not args.no_transients:
            plot_transient_results(results_dir, args.transient_currents)

        print(f"\n🎉 Analysis complete! Results from: {csv_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
