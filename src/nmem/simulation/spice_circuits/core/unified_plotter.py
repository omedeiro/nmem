#!/usr/bin/env python3
"""
Unified Plotting System for SPICE Simulation Results

This module provides a comprehensive, modular plotting system for SPICE simulation
results. It consolidates all the existing plotting functionality into a single,
well-organized interface.

Features:
- Modular plot types (transient, sweep, comparison, etc.)
- Flexible configuration system
- Support for multiple simulation result formats
- Automatic plot styling and formatting
- Export capabilities (PNG, PDF, SVG)
- Interactive plotting options

Usage:
    from nmem.simulation.spice_circuits.core.unified_plotter import UnifiedPlotter

    plotter = UnifiedPlotter(config_path="plotting_config.yaml")
    plotter.plot_transient_analysis(data, output_path="results.png")
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
from matplotlib.patches import Rectangle
from cycler import cycler

# Import simulation data processing functions
try:
    from ..functions import get_step_parameter, filter_first
    from ..config.settings import PlottingConfig
except ImportError:
    # Fallback for standalone usage
    def get_step_parameter(data_dict):
        """Fallback function to get step parameter."""
        for key in ["read_current", "write_current", "enable_current"]:
            if key in data_dict:
                return key
        return "time"

    def filter_first(data):
        """Fallback function to filter first element."""
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            return data[0]
        return data


class PlotType(Enum):
    """Enumeration of available plot types."""

    TRANSIENT = "transient"
    CURRENT_SWEEP = "current_sweep"
    VOLTAGE_OUTPUT = "voltage_output"
    BER_ANALYSIS = "ber_analysis"
    SWITCHING_PROBABILITY = "switching_probability"
    PERSISTENT_CURRENT = "persistent_current"
    WAVEFORM_COMPARISON = "waveform_comparison"
    MULTI_PANEL = "multi_panel"
    CUSTOM = "custom"


class PlotStyle(Enum):
    """Enumeration of plot styles."""

    PUBLICATION = "publication"
    PRESENTATION = "presentation"
    TECHNICAL = "technical"
    MINIMAL = "minimal"
    CUSTOM = "custom"


@dataclass
class PlotConfig:
    """Configuration for individual plots."""

    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 300
    font_size: int = 12
    line_width: float = 1.5
    marker_size: float = 6
    grid: bool = True
    grid_alpha: float = 0.3
    legend: bool = True
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    colormap: str = "coolwarm"
    style: PlotStyle = PlotStyle.TECHNICAL
    transparent: bool = False


@dataclass
class ExportConfig:
    """Configuration for plot export."""

    format: str = "png"  # png, pdf, svg, eps
    dpi: int = 300
    bbox_inches: str = "tight"
    transparent: bool = False
    optimize: bool = True


class UnifiedPlotter:
    """
    Unified plotting system for SPICE simulation results.

    This class provides a modular interface for creating various types of plots
    from SPICE simulation data with consistent styling and formatting.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the unified plotter.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.plot_config = PlotConfig()
        self.export_config = ExportConfig()

        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

        # Initialize plotting style
        self._setup_matplotlib()

        # Color schemes
        self.colors = {
            "left": "#1f77b4",  # Blue
            "right": "#d62728",  # Red
            "voltage": "#9467bd",  # Purple
            "enable": "#ff7f0e",  # Orange
            "critical": "#2ca02c",  # Green
        }

        # Plot type handlers
        self.plot_handlers = {
            PlotType.TRANSIENT: self._plot_transient,
            PlotType.CURRENT_SWEEP: self._plot_current_sweep,
            PlotType.VOLTAGE_OUTPUT: self._plot_voltage_output,
            PlotType.BER_ANALYSIS: self._plot_ber_analysis,
            PlotType.SWITCHING_PROBABILITY: self._plot_switching_probability,
            PlotType.PERSISTENT_CURRENT: self._plot_persistent_current,
            PlotType.WAVEFORM_COMPARISON: self._plot_waveform_comparison,
            PlotType.MULTI_PANEL: self._plot_multi_panel,
            PlotType.CUSTOM: self._plot_custom,
        }

    def extract_simulation_data(self, ltspice_data) -> Dict[str, Any]:
        """Extract and standardize data from LTspice simulation results.

        Args:
            ltspice_data: Parsed LTspice data object

        Returns:
            Dictionary containing standardized simulation data
        """
        import numpy as np

        try:
            # Get time data
            time = ltspice_data.get_time()

            # Map LTspice variable names to standard names
            variable_mapping = {
                "Ix(hl:drain)": "left_current",
                "Ix(hr:drain)": "right_current",
                "Ix(HL:drain)": "left_current",
                "Ix(HR:drain)": "right_current",
                "V(out)": "output_voltage",
                "I(R1)": "enable_current",
                "I(R2)": "input_current",
                "Ix(hl:heater_p)": "left_heater_current",
                "Ix(hr:heater_p)": "right_heater_current",
                "I(ichl)": "left_critical_current",
                "I(ichr)": "right_critical_current",
            }

            data = {"time": time}

            # Get available variables
            if hasattr(ltspice_data, "get_variable_names"):
                available_vars = ltspice_data.get_variable_names()
            else:
                available_vars = ltspice_data.getVariableNames()  # deprecated method

            # Extract available signals with mapping
            signals_found = 0
            for ltspice_var, standard_name in variable_mapping.items():
                if ltspice_var in available_vars:
                    try:
                        signal = ltspice_data.get_data(ltspice_var)
                        data[standard_name] = signal
                        signals_found += 1
                        print(f"   ✅ Mapped {ltspice_var} → {standard_name}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to extract {ltspice_var}: {e}")

            # Add any unmapped variables with their original names
            for var in available_vars:
                if (
                    var not in variable_mapping
                    and var != "time"
                    and not var.startswith("V(")
                ):
                    try:
                        signal = ltspice_data.get_data(var)
                        clean_name = (
                            var.replace("(", "_")
                            .replace(")", "")
                            .replace(":", "_")
                            .lower()
                        )
                        data[clean_name] = signal
                        signals_found += 1
                        print(f"   ✅ Added {var} → {clean_name}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to extract {var}: {e}")

            print(f"✅ Extracted {signals_found} signals from simulation")
            return data

        except Exception as e:
            print(f"Failed to extract simulation data: {e}")
            return {"time": np.array([0, 1]), "error": True}

    def _setup_matplotlib(self):
        """Setup matplotlib with consistent styling."""
        # Set up color cycle
        plt.rcParams.update(
            {
                "axes.prop_cycle": cycler(
                    color=[
                        "#1f77b4",
                        "#d62728",
                        "#2ca02c",
                        "#ff7f0e",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf",
                    ]
                ),
                "font.size": self.plot_config.font_size,
                "lines.linewidth": self.plot_config.line_width,
                "lines.markersize": self.plot_config.marker_size,
                "grid.alpha": self.plot_config.grid_alpha,
                "figure.dpi": self.plot_config.dpi,
                "savefig.dpi": self.plot_config.dpi,
                "savefig.transparent": self.plot_config.transparent,
                "savefig.bbox": "tight",
            }
        )

        # Apply style-specific settings
        if self.plot_config.style == PlotStyle.PUBLICATION:
            plt.rcParams.update(
                {
                    "font.size": 10,
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "lines.linewidth": 1.0,
                }
            )
        elif self.plot_config.style == PlotStyle.PRESENTATION:
            plt.rcParams.update(
                {
                    "font.size": 14,
                    "axes.labelsize": 16,
                    "axes.titlesize": 18,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.fontsize": 12,
                    "lines.linewidth": 2.0,
                }
            )

    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            print(f"Warning: Config file {config_path} not found, using defaults")
            return

        try:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                import yaml

                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                with open(config_path, "r") as f:
                    config_data = json.load(f)
            else:
                print(f"Warning: Unsupported config format {config_path.suffix}")
                return

            # Update configurations
            if "plot" in config_data:
                for key, value in config_data["plot"].items():
                    if hasattr(self.plot_config, key):
                        setattr(self.plot_config, key, value)

            if "export" in config_data:
                for key, value in config_data["export"].items():
                    if hasattr(self.export_config, key):
                        setattr(self.export_config, key, value)

        except Exception as e:
            print(f"Warning: Error loading config: {e}")

    def create_plot(
        self,
        plot_type: Union[PlotType, str],
        data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Create a plot of the specified type.

        Args:
            plot_type: Type of plot to create
            data: Simulation data dictionary
            output_path: Optional path to save the plot
            show: Whether to display the plot
            **kwargs: Additional plot-specific arguments

        Returns:
            Matplotlib figure object
        """
        if isinstance(plot_type, str):
            plot_type = PlotType(plot_type)

        # Get plot handler
        handler = self.plot_handlers.get(plot_type)
        if not handler:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        # Create the plot
        fig = handler(data, **kwargs)

        # Save if requested
        if output_path:
            self.save_plot(fig, output_path)

        # Show if requested
        if show:
            plt.show()

        return fig

    def _plot_transient(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot transient analysis results."""
        cases = kwargs.get("cases", [0])
        signals = kwargs.get("signals", ["left_current", "right_current"])
        time_window = kwargs.get("time_window", None)

        fig, ax = plt.subplots(
            figsize=self.plot_config.figsize, dpi=self.plot_config.dpi
        )

        for case in cases:
            case_data = data[case] if isinstance(data, dict) and case in data else data

            time = case_data.get("time", case_data.get("tran_time", np.array([])))

            # Apply time window if specified
            if time_window:
                mask = (time >= time_window[0]) & (time <= time_window[1])
                time = time[mask]
            else:
                mask = slice(None)

            # Auto-detect available signals if default ones aren't found
            available_signals = []
            for signal in signals:
                if signal in case_data:
                    available_signals.append(signal)

            # If no default signals found, try to find current/voltage signals
            if not available_signals:
                for key in case_data.keys():
                    if (
                        any(
                            keyword in key.lower() for keyword in ["current", "voltage"]
                        )
                        and key != "time"
                    ):
                        available_signals.append(key)
                        if len(available_signals) >= 6:  # Limit to avoid cluttered plot
                            break

            # Plot each available signal
            plotted_any = False
            ylabel = "Signal"
            for i, signal in enumerate(available_signals):
                signal_data = case_data.get(signal, np.array([]))
                if len(signal_data) > 0:
                    label = signal.replace("_", " ").title()
                    if len(cases) > 1:
                        label += f" (Case {case})"

                    # Determine appropriate scaling and units
                    if "current" in signal.lower():
                        signal_data = signal_data * 1e6  # Convert to µA
                        ylabel = "Current (μA)"
                    elif "voltage" in signal.lower():
                        signal_data = signal_data * 1e3  # Convert to mV
                        ylabel = "Voltage (mV)"
                    else:
                        ylabel = signal.replace("_", " ").title()

                    # Choose color
                    if "left" in signal.lower():
                        color = self.colors.get("left", None)
                    elif "right" in signal.lower():
                        color = self.colors.get("right", None)
                    elif "voltage" in signal.lower():
                        color = self.colors.get("voltage", None)
                    else:
                        color = None

                    ax.plot(
                        time * 1e9,
                        signal_data,
                        label=label,
                        color=color,
                        linewidth=self.plot_config.line_width,
                    )
                    plotted_any = True

            # If no data was plotted, add a message
            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "No data available to plot",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )

        # Formatting
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(ylabel if "ylabel" in locals() else "Signal")
        if self.plot_config.grid:
            ax.grid(True, alpha=self.plot_config.grid_alpha)
        if self.plot_config.legend and available_signals and plotted_any:
            ax.legend()

        # Apply custom limits or auto-scale properly
        if self.plot_config.xlim:
            ax.set_xlim(self.plot_config.xlim)
        else:
            # Auto-scale time axis
            if len(time) > 0:
                time_ns = time * 1e9
                ax.set_xlim([np.min(time_ns), np.max(time_ns)])

        if self.plot_config.ylim:
            ax.set_ylim(self.plot_config.ylim)
        # Let matplotlib auto-scale Y axis for better visibility

        if self.plot_config.title:
            ax.set_title(self.plot_config.title)

        plt.tight_layout()
        return fig

    def _plot_current_sweep(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot current sweep analysis."""
        sweep_type = kwargs.get("sweep_type", "read_current")
        metrics = kwargs.get("metrics", ["bit_error_rate", "switching_probability"])

        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=self.plot_config.figsize,
            dpi=self.plot_config.dpi,
            sharex=True,
        )
        if len(metrics) == 1:
            axes = [axes]

        # Process data
        case_data = data[0] if isinstance(data, dict) and 0 in data else data
        sweep_param = get_step_parameter(case_data)
        sweep_values = case_data.get(sweep_param, np.array([]))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            metric_data = case_data.get(metric, np.array([]))

            if len(metric_data) > 0 and len(sweep_values) > 0:
                ax.plot(
                    sweep_values * 1e6,
                    metric_data,
                    "o-",
                    linewidth=self.plot_config.line_width,
                    markersize=self.plot_config.marker_size,
                )

                ax.set_ylabel(metric.replace("_", " ").title())
                if self.plot_config.grid:
                    ax.grid(True, alpha=self.plot_config.grid_alpha)

        axes[-1].set_xlabel(f'{sweep_param.replace("_", " ").title()} (μA)')

        if self.plot_config.title:
            fig.suptitle(self.plot_config.title)

        plt.tight_layout()
        return fig

    def _plot_voltage_output(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot voltage output analysis."""
        time_window = kwargs.get("time_window", None)

        fig, ax = plt.subplots(
            figsize=self.plot_config.figsize, dpi=self.plot_config.dpi
        )

        case_data = data[0] if isinstance(data, dict) and 0 in data else data
        time = case_data.get("time", case_data.get("tran_time", np.array([])))

        # Try different voltage signal names
        voltage_signals = [
            "output_voltage",
            "voltage_out",
            "tran_output_voltage",
            "V(out)",
            "vout",
        ]
        voltage = None
        voltage_name = None

        for sig_name in voltage_signals:
            if sig_name in case_data:
                voltage = case_data[sig_name]
                voltage_name = sig_name
                break

        # If no specific voltage found, find any voltage signal
        if voltage is None:
            for key, val in case_data.items():
                if "voltage" in key.lower() and key != "time" and len(val) > 0:
                    voltage = val
                    voltage_name = key
                    break

        if voltage is None or len(voltage) == 0:
            # Create empty plot with message
            ax.text(
                0.5,
                0.5,
                "No voltage data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Voltage (mV)")
            plt.tight_layout()
            return fig

        # Apply time window if specified
        if time_window:
            mask = (time >= time_window[0]) & (time <= time_window[1])
            time = time[mask]
            voltage = voltage[mask]

        if len(voltage) > 0:
            ax.plot(
                time * 1e9,
                voltage * 1e3,
                color=self.colors["voltage"],
                linewidth=self.plot_config.line_width,
                label=f"Output Voltage ({voltage_name})",
            )

            # Add zero reference line
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.7)

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Voltage (mV)")
        if self.plot_config.grid:
            ax.grid(True, alpha=self.plot_config.grid_alpha)
        if self.plot_config.legend:
            ax.legend()

        plt.tight_layout()
        return fig

    def _plot_ber_analysis(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot bit error rate analysis."""
        return self._plot_current_sweep(data, metrics=["bit_error_rate"], **kwargs)

    def _plot_switching_probability(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot switching probability analysis."""
        return self._plot_current_sweep(
            data, metrics=["switching_probability"], **kwargs
        )

    def _plot_persistent_current(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot persistent current analysis."""
        return self._plot_current_sweep(data, metrics=["persistent_current"], **kwargs)

    def _plot_waveform_comparison(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Plot comparison of different waveforms."""
        waveform_configs = kwargs.get("waveform_configs", ["config1", "config2"])

        fig, axes = plt.subplots(
            2,
            1,
            figsize=self.plot_config.figsize,
            dpi=self.plot_config.dpi,
            sharex=True,
        )

        # Plot current comparison
        for i, config in enumerate(waveform_configs):
            if config in data:
                config_data = data[config]
                time = config_data.get("time", np.array([])) * 1e9
                left_current = config_data.get("left_current", np.array([])) * 1e6
                right_current = config_data.get("right_current", np.array([])) * 1e6

                axes[0].plot(
                    time,
                    left_current,
                    label=f"{config} - Left",
                    color=self.colors["left"],
                    alpha=0.7 + 0.3 * i,
                )
                axes[0].plot(
                    time,
                    right_current,
                    label=f"{config} - Right",
                    color=self.colors["right"],
                    alpha=0.7 + 0.3 * i,
                )

        # Plot voltage comparison
        for i, config in enumerate(waveform_configs):
            if config in data:
                config_data = data[config]
                time = config_data.get("time", np.array([])) * 1e9
                voltage = config_data.get("output_voltage", np.array([])) * 1e3

                axes[1].plot(
                    time,
                    voltage,
                    label=f"{config} - Voltage",
                    color=self.colors["voltage"],
                    alpha=0.7 + 0.3 * i,
                )

        axes[0].set_ylabel("Current (μA)")
        axes[1].set_ylabel("Voltage (mV)")
        axes[1].set_xlabel("Time (ns)")

        for ax in axes:
            if self.plot_config.grid:
                ax.grid(True, alpha=self.plot_config.grid_alpha)
            if self.plot_config.legend:
                ax.legend()

        plt.tight_layout()
        return fig

    def _plot_multi_panel(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create multi-panel plot with different analysis views."""
        panel_config = kwargs.get(
            "panel_config",
            {
                "transient": {"row": 0, "col": 0},
                "voltage": {"row": 1, "col": 0},
                "sweep": {"row": 0, "col": 1, "rowspan": 2},
            },
        )

        rows = max(
            config.get("row", 0) + config.get("rowspan", 1)
            for config in panel_config.values()
        )
        cols = max(
            config.get("col", 0) + config.get("colspan", 1)
            for config in panel_config.values()
        )

        fig = plt.figure(
            figsize=(
                self.plot_config.figsize[0] * cols,
                self.plot_config.figsize[1] * rows,
            ),
            dpi=self.plot_config.dpi,
        )
        gs = gridspec.GridSpec(rows, cols, figure=fig)

        axes = {}

        for panel_type, config in panel_config.items():
            row = config.get("row", 0)
            col = config.get("col", 0)
            rowspan = config.get("rowspan", 1)
            colspan = config.get("colspan", 1)

            ax = fig.add_subplot(gs[row : row + rowspan, col : col + colspan])
            axes[panel_type] = ax

            # Create subplot based on type
            if panel_type == "transient":
                self._plot_transient_on_axis(ax, data, **kwargs)
            elif panel_type == "voltage":
                self._plot_voltage_on_axis(ax, data, **kwargs)
            elif panel_type == "sweep":
                self._plot_sweep_on_axis(ax, data, **kwargs)

        plt.tight_layout()
        return fig

    def _plot_transient_on_axis(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        """Plot transient data on given axis."""
        case_data = data[0] if isinstance(data, dict) and 0 in data else data
        time = case_data.get("time", np.array([])) * 1e9
        left_current = case_data.get("left_current", np.array([])) * 1e6
        right_current = case_data.get("right_current", np.array([])) * 1e6

        if len(time) > 0:
            ax.plot(time, left_current, label="Left Current", color=self.colors["left"])
            ax.plot(
                time, right_current, label="Right Current", color=self.colors["right"]
            )

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Current (μA)")
        ax.legend()
        ax.grid(True, alpha=self.plot_config.grid_alpha)

    def _plot_voltage_on_axis(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        """Plot voltage data on given axis."""
        case_data = data[0] if isinstance(data, dict) and 0 in data else data
        time = case_data.get("time", np.array([])) * 1e9
        voltage = case_data.get("output_voltage", np.array([])) * 1e3

        if len(time) > 0:
            ax.plot(time, voltage, color=self.colors["voltage"])
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Voltage (mV)")
        ax.grid(True, alpha=self.plot_config.grid_alpha)

    def _plot_sweep_on_axis(self, ax: plt.Axes, data: Dict[str, Any], **kwargs):
        """Plot sweep data on given axis."""
        case_data = data[0] if isinstance(data, dict) and 0 in data else data
        sweep_param = get_step_parameter(case_data)
        sweep_values = case_data.get(sweep_param, np.array([])) * 1e6
        ber = case_data.get("bit_error_rate", np.array([]))

        if len(sweep_values) > 0 and len(ber) > 0:
            ax.plot(sweep_values, ber, "o-")

        ax.set_xlabel(f'{sweep_param.replace("_", " ").title()} (μA)')
        ax.set_ylabel("BER")
        ax.grid(True, alpha=self.plot_config.grid_alpha)

    def _plot_custom(self, data: Dict[str, Any], **kwargs) -> plt.Figure:
        """Create custom plot based on user-provided function."""
        custom_func = kwargs.get("custom_function")
        if not custom_func:
            raise ValueError("Custom plot requires 'custom_function' parameter")

        fig = plt.figure(figsize=self.plot_config.figsize, dpi=self.plot_config.dpi)
        return custom_func(fig, data, **kwargs)

    def save_plot(self, fig: plt.Figure, output_path: Union[str, Path]):
        """Save plot to file with configured export settings."""
        output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format from extension or use configured format
        file_format = output_path.suffix.lstrip(".") or self.export_config.format

        # Prepare save arguments
        save_kwargs = {
            "format": file_format,
            "dpi": self.export_config.dpi,
            "bbox_inches": self.export_config.bbox_inches,
            "transparent": self.export_config.transparent,
        }

        # Only add optimize for formats that support it
        if file_format.lower() in ["pdf", "eps"]:
            save_kwargs["optimize"] = self.export_config.optimize

        # Save with configured settings
        fig.savefig(output_path, **save_kwargs)

        print(f"Plot saved to: {output_path}")

    def create_animation(
        self, data: Dict[str, Any], output_path: Union[str, Path], **kwargs
    ) -> str:
        """Create animation from time series data."""
        try:
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("Animation requires matplotlib.animation")

        # This is a placeholder for animation functionality
        # Implementation would depend on specific requirements
        print("Animation functionality not yet implemented")
        return str(output_path)

    def generate_report(
        self, data: Dict[str, Any], output_dir: Union[str, Path], **kwargs
    ) -> Path:
        """Generate comprehensive plotting report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create different plot types
        plot_configs = [
            (PlotType.TRANSIENT, "transient_analysis.png"),
            (PlotType.VOLTAGE_OUTPUT, "voltage_output.png"),
            (PlotType.CURRENT_SWEEP, "current_sweep.png"),
            (PlotType.MULTI_PANEL, "multi_panel_analysis.png"),
        ]

        created_plots = []

        for plot_type, filename in plot_configs:
            try:
                output_path = output_dir / filename
                fig = self.create_plot(plot_type, data, output_path, show=False)
                plt.close(fig)  # Close to save memory
                created_plots.append(output_path)
                print(f"Created: {filename}")
            except Exception as e:
                print(f"Warning: Could not create {filename}: {e}")

        print(f"Report generated in: {output_dir}")
        print(f"Created {len(created_plots)} plots")

        return output_dir


def create_default_plotting_config(output_path: Union[str, Path]) -> Dict[str, Any]:
    """Create a default plotting configuration file."""
    config = {
        "plot": asdict(PlotConfig()),
        "export": asdict(ExportConfig()),
        "colors": {
            "left": "#1f77b4",
            "right": "#d62728",
            "voltage": "#9467bd",
            "enable": "#ff7f0e",
            "critical": "#2ca02c",
        },
    }

    output_path = Path(output_path)

    if output_path.suffix.lower() in [".yaml", ".yml"]:
        import yaml

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    else:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    return config


# Convenience functions for quick plotting
def quick_transient_plot(
    data: Dict[str, Any], output_path: Optional[str] = None, show: bool = True
) -> plt.Figure:
    """Quick function to create a transient plot."""
    plotter = UnifiedPlotter()
    return plotter.create_plot(PlotType.TRANSIENT, data, output_path, show)


def quick_sweep_plot(
    data: Dict[str, Any], output_path: Optional[str] = None, show: bool = True
) -> plt.Figure:
    """Quick function to create a current sweep plot."""
    plotter = UnifiedPlotter()
    return plotter.create_plot(PlotType.CURRENT_SWEEP, data, output_path, show)


def quick_voltage_plot(
    data: Dict[str, Any], output_path: Optional[str] = None, show: bool = True
) -> plt.Figure:
    """Quick function to create a voltage output plot."""
    plotter = UnifiedPlotter()
    return plotter.create_plot(PlotType.VOLTAGE_OUTPUT, data, output_path, show)


if __name__ == "__main__":
    # Example usage and testing
    print("Unified Plotter for SPICE Simulation Results")
    print("============================================")

    # Create default config
    config_path = Path("default_plotting_config.yaml")
    if not config_path.exists():
        create_default_plotting_config(config_path)
        print(f"Created default config: {config_path}")

    # Example data structure (for testing)
    example_data = {
        0: {
            "time": np.linspace(0, 1e-6, 1000),
            "left_current": np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-6, 1000))
            * 100e-6,
            "right_current": np.cos(2 * np.pi * 1e6 * np.linspace(0, 1e-6, 1000))
            * 100e-6,
            "output_voltage": np.random.randn(1000) * 10e-3,
            "read_current": np.linspace(600e-6, 900e-6, 100),
            "bit_error_rate": np.random.rand(100),
        }
    }

    # Test plotter
    plotter = UnifiedPlotter(config_path)

    print("Testing unified plotter...")
    print("Available plot types:")
    for plot_type in PlotType:
        print(f"  - {plot_type.value}")
