#!/usr/bin/env python3
"""Unified waveform generation script - replaces all create_waveform*.py scripts."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union

from nmem.simulation.spice_circuits.waveform.generators import WaveformGenerator
from nmem.simulation.spice_circuits.utils.file_io import (
    save_pwl_file,
    get_waveforms_dir,
    get_data_dir,
)
from nmem.simulation.spice_circuits.config.settings import (
    ConfigManager,
    get_default_config,
)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified waveform generation for SPICE simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard waveforms with descriptive names
  python unified_waveform_generator.py --preset standard
  # → Creates: standard_waveform_chan.txt, standard_waveform_enab.txt
  
  # Generate fast waveforms with descriptive names
  python unified_waveform_generator.py --preset fast
  # → Creates: fast_waveform_chan.txt, fast_waveform_enab.txt
  
  # Generate with legacy filenames for backward compatibility
  python unified_waveform_generator.py --preset standard --legacy-names
  # → Creates: chan.txt, enab.txt
  
  # Parameter sweep (creates organized directory structure)
  python unified_waveform_generator.py --sweep read_amplitude --values 715e-6,720e-6,725e-6
  
  # Custom waveform with specific name
  python unified_waveform_generator.py --cycle-time 500e-9 --output-name test_timing
        """,
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        "-p",
        choices=["standard", "fast"],
        help="Use preset configuration (standard=default timing, fast=faster timing)",
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (YAML or JSON)"
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for waveform files"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="waveform",
        help="Base name for output files (default: waveform → waveform_chan.txt, waveform_enab.txt)",
    )

    parser.add_argument(
        "--legacy-names",
        action="store_true",
        help="Use legacy filenames (chan.txt, enab.txt) for backward compatibility",
    )

    # Sweep options
    parser.add_argument(
        "--sweep",
        choices=[
            "read_amplitude",
            "write_amplitude",
            "cycle_time",
            "enab_write_amplitude",
        ],
        help="Parameter to sweep",
    )

    parser.add_argument(
        "--values",
        type=str,
        help="Comma-separated sweep values (e.g., '715e-6,720e-6,725e-6')",
    )

    parser.add_argument(
        "--sweep-range",
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Sweep range as start,stop,step (e.g., 715e-6 730e-6 5e-6)",
    )

    # Waveform parameter overrides
    parser.add_argument("--cycle-time", type=float, help="Cycle time (s)")
    parser.add_argument("--pulse-sigma", type=float, help="Pulse sigma (s)")
    parser.add_argument("--write-amplitude", type=float, help="Write amplitude (A)")
    parser.add_argument("--read-amplitude", type=float, help="Read amplitude (A)")
    parser.add_argument(
        "--enab-write-amplitude", type=float, help="Enable write amplitude (A)"
    )
    parser.add_argument(
        "--enab-read-amplitude", type=float, help="Enable read amplitude (A)"
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Generate and display plot of waveforms"
    )

    parser.add_argument(
        "--save-plot", type=str, help="Save plot to file (specify filename)"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print operation summary (default: True)",
    )

    return parser


def load_preset_config(preset: str):
    """Load a preset configuration."""
    if preset == "standard":
        config_file = Path(__file__).parent.parent / "config" / "standard_waveform.yaml"
    elif preset == "fast":
        config_file = Path(__file__).parent.parent / "config" / "fast_waveform.yaml"
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if config_file.exists():
        return ConfigManager.load_config(config_file)
    else:
        print(f"Warning: Preset config file not found: {config_file}")
        return get_default_config()


def create_generator_from_config(config, args):
    """Create WaveformGenerator with parameters from config and args."""
    return WaveformGenerator(
        cycle_time=args.cycle_time or config.waveform.cycle_time,
        pulse_sigma=args.pulse_sigma or config.waveform.pulse_sigma,
        hold_width_write=config.waveform.hold_width_write,
        hold_width_read=config.waveform.hold_width_read,
        hold_width_clear=config.waveform.hold_width_clear,
        write_amplitude=args.write_amplitude or config.waveform.write_amplitude,
        read_amplitude=args.read_amplitude or config.waveform.read_amplitude,
        enab_write_amplitude=args.enab_write_amplitude
        or config.waveform.enab_write_amplitude,
        enab_read_amplitude=args.enab_read_amplitude
        or config.waveform.enab_read_amplitude,
        clear_amplitude=config.waveform.clear_amplitude,
        dt=config.waveform.dt,
    )


def generate_single_waveform(generator: WaveformGenerator, args, output_dir: Path):
    """Generate a single waveform pair."""
    print("Generating waveform...")

    t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
        generator.generate_memory_protocol_sequence(seed=args.seed)
    )

    # Generate filenames based on options
    if args.legacy_names:
        # Use legacy names for backward compatibility
        chan_file = "chan.txt"
        enab_file = "enab.txt"
    else:
        # Generate descriptive filenames based on configuration
        if args.preset:
            prefix = f"{args.preset}_{args.output_name}"
        else:
            prefix = args.output_name

        chan_file = f"{prefix}_chan.txt"
        enab_file = f"{prefix}_enab.txt"

    # Save files
    chan_path = output_dir / chan_file
    enab_path = output_dir / enab_file

    save_pwl_file(chan_path, t_chan, i_chan)
    save_pwl_file(enab_path, t_enab, i_enab)

    print(f"✅ Waveforms saved to: {output_dir}")
    print(f"   - Channel: {chan_path}")
    print(f"   - Enable:  {enab_path}")

    return t_chan, i_chan, t_enab, i_enab, ops, enab_on


def generate_parameter_sweep(generator: WaveformGenerator, args, output_dir: Path):
    """Generate a parameter sweep."""
    if not args.sweep:
        raise ValueError("Sweep parameter not specified")

    # Parse sweep values
    if args.values:
        sweep_values = [float(v) for v in args.values.split(",")]
    elif args.sweep_range:
        start, stop, step = [float(v) for v in args.sweep_range]
        sweep_values = np.arange(start, stop + step / 2, step)  # Include endpoint
    else:
        raise ValueError(
            "Either --values or --sweep-range must be specified for sweeps"
        )

    print(f"Generating {args.sweep} sweep with {len(sweep_values)} values...")

    # Create sweep subdirectory
    sweep_dir = output_dir / f"{args.sweep}_sweep"
    sweep_dir.mkdir(exist_ok=True)

    pwl_files = []

    for i, value in enumerate(sweep_values):
        # Update the generator parameter
        if args.sweep == "read_amplitude":
            generator.read_amplitude = value
        elif args.sweep == "write_amplitude":
            generator.write_amplitude = value
        elif args.sweep == "cycle_time":
            generator.cycle_time = value
        elif args.sweep == "enab_write_amplitude":
            generator.enab_write_amplitude = value

        # Generate waveform
        t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
            generator.generate_memory_protocol_sequence(seed=args.seed)
        )

        # Save to file
        if args.sweep in ["read_amplitude", "write_amplitude", "enab_write_amplitude"]:
            filename = sweep_dir / f"{args.sweep}_{int(value * 1e6):+05g}u.txt"
        else:
            filename = sweep_dir / f"{args.sweep}_{value:.2e}.txt"

        save_pwl_file(filename, t_chan, i_chan)
        pwl_files.append(str(filename))

        print(f"   Generated {i+1}/{len(sweep_values)}: {filename.name}")

    # Save sweep summary
    df = pd.DataFrame(
        {
            f"{args.sweep.replace('_', ' ').title()}": sweep_values,
            "Waveform File": pwl_files,
        }
    )

    summary_file = sweep_dir / "sweep_summary.csv"
    df.to_csv(summary_file, index=False)

    print(f"✅ Sweep completed: {len(sweep_values)} files generated")
    print(f"   Sweep directory: {sweep_dir}")
    print(f"   Summary saved to: {summary_file}")


def plot_waveforms(t_chan, i_chan, t_enab, i_enab, save_path=None, show=True):
    """Plot the generated waveforms."""
    import matplotlib.pyplot as plt

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
    if args.preset:
        config = load_preset_config(args.preset)
        print(f"Using preset: {args.preset}")
    elif args.config:
        config = ConfigManager.load_config(args.config)
        print(f"Using config: {args.config}")
    else:
        config = get_default_config()
        print("Using default configuration")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.sweep:
        output_dir = get_data_dir()
    else:
        output_dir = get_waveforms_dir()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = create_generator_from_config(config, args)

    # Generate waveforms
    if args.sweep:
        generate_parameter_sweep(generator, args, output_dir)

        # For sweeps, we don't plot or show operation summary
        return
    else:
        t_chan, i_chan, t_enab, i_enab, ops, enab_on = generate_single_waveform(
            generator, args, output_dir
        )

    # Print operation summary
    if args.summary:
        print_operation_summary(ops, enab_on)

    # Plot if requested
    if args.plot or args.save_plot:
        plot_waveforms(
            t_chan, i_chan, t_enab, i_enab, save_path=args.save_plot, show=args.plot
        )

    print("\n✅ Waveform generation completed successfully!")


if __name__ == "__main__":
    main()
