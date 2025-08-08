#!/usr/bin/env python3
"""Script for generating memory protocol waveforms."""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from nmem.simulation.spice_circuits.waveform.generators import WaveformGenerator
from nmem.simulation.spice_circuits.utils.file_io import (
    save_pwl_file,
    get_waveforms_dir,
)
from nmem.simulation.spice_circuits.config.settings import (
    ConfigManager,
    get_default_config,
)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate memory protocol waveforms for SPICE simulation"
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (YAML or JSON)"
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for waveform files"
    )

    parser.add_argument(
        "--chan-file",
        type=str,
        default="chan.txt",
        help="Output filename for channel waveform (default: chan.txt)",
    )

    parser.add_argument(
        "--enab-file",
        type=str,
        default="enab.txt",
        help="Output filename for enable waveform (default: enab.txt)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Generate and display plot of waveforms",
    )

    parser.add_argument(
        "--save-plot", type=str, help="Save plot to file (specify filename)"
    )

    # Waveform parameter overrides
    parser.add_argument("--cycle-time", type=float, help="Cycle time (s)")
    parser.add_argument("--pulse-sigma", type=float, help="Pulse sigma (s)")
    parser.add_argument("--write-amplitude", type=float, help="Write amplitude (A)")
    parser.add_argument("--read-amplitude", type=float, help="Read amplitude (A)")

    return parser


def generate_waveforms(config, args):
    """Generate waveforms using configuration and command line arguments."""

    # Create waveform generator with config parameters
    generator = WaveformGenerator(
        cycle_time=args.cycle_time or config.waveform.cycle_time,
        pulse_sigma=args.pulse_sigma or config.waveform.pulse_sigma,
        hold_width_write=config.waveform.hold_width_write,
        hold_width_read=config.waveform.hold_width_read,
        hold_width_clear=config.waveform.hold_width_clear,
        write_amplitude=args.write_amplitude or config.waveform.write_amplitude,
        read_amplitude=args.read_amplitude or config.waveform.read_amplitude,
        enab_write_amplitude=config.waveform.enab_write_amplitude,
        enab_read_amplitude=config.waveform.enab_read_amplitude,
        clear_amplitude=config.waveform.clear_amplitude,
        dt=config.waveform.dt,
    )

    # Generate waveforms
    t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
        generator.generate_memory_protocol_sequence(seed=args.seed)
    )

    return t_chan, i_chan, t_enab, i_enab, ops, enab_on


def save_waveforms(t_chan, i_chan, t_enab, i_enab, output_dir, chan_file, enab_file):
    """Save waveforms to PWL files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chan_path = output_dir / chan_file
    enab_path = output_dir / enab_file

    save_pwl_file(chan_path, t_chan, i_chan)
    save_pwl_file(enab_path, t_enab, i_enab)

    print(f"Saved channel waveform to: {chan_path}")
    print(f"Saved enable waveform to: {enab_path}")

    return chan_path, enab_path


def plot_waveforms(t_chan, i_chan, t_enab, i_enab, save_path=None, show=True):
    """Plot the generated waveforms."""

    plt.figure(figsize=(12, 6))
    plt.plot(
        t_chan * 1e9,
        i_chan * 1e6,
        label="I_chan (data)",
        color="tab:blue",
        linewidth=1.5,
    )
    plt.plot(
        t_enab * 1e9,
        i_enab * 1e6,
        label="I_enab (word-line)",
        color="tab:orange",
        linewidth=1.5,
    )
    plt.xlabel("Time (ns)")
    plt.ylabel("Current (µA)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()


def print_operation_summary(ops, enab_on, max_ops=10):
    """Print summary of operations."""
    print("\nOperation Summary:")
    print("Slot | Operation | Wordline Active")
    print("-" * 35)
    for i in range(min(max_ops, len(ops))):
        print(f"{i:>4} | {ops[i]:>9} | {'ON' if enab_on[i] else 'OFF'}")

    if len(ops) > max_ops:
        print(f"... ({len(ops) - max_ops} more operations)")


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        config = get_default_config()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_waveforms_dir()

    # Generate waveforms
    print("Generating memory protocol waveforms...")
    t_chan, i_chan, t_enab, i_enab, ops, enab_on = generate_waveforms(config, args)

    # Save waveforms
    chan_path, enab_path = save_waveforms(
        t_chan, i_chan, t_enab, i_enab, output_dir, args.chan_file, args.enab_file
    )

    # Print operation summary
    print_operation_summary(ops, enab_on)

    # Plot if requested
    if args.plot or args.save_plot:
        plot_waveforms(
            t_chan, i_chan, t_enab, i_enab, save_path=args.save_plot, show=args.plot
        )

    print("\nWaveform generation completed successfully!")


if __name__ == "__main__":
    main()
