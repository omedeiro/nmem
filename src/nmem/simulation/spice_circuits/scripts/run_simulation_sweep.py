#!/usr/bin/env python3
"""Script for automated SPICE simulation sweeps."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from nmem.simulation.spice_circuits.core.ltspice_interface import (
    LTspiceRunner,
    SimulationAutomator,
)
from nmem.simulation.spice_circuits.config.settings import (
    ConfigManager,
    get_default_config,
)
from nmem.simulation.spice_circuits.utils.file_io import get_data_dir, get_results_dir


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("spice_automation.log")],
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Automate SPICE simulation parameter sweeps"
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration file (YAML or JSON)"
    )

    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to SPICE netlist template file",
    )

    parser.add_argument(
        "--wave-dir",
        type=str,
        required=True,
        help="Directory containing PWL waveform files",
    )

    parser.add_argument("--output", "-o", type=str, help="Output CSV file for results")

    parser.add_argument(
        "--ltspice-path", type=str, help="Path to LTspice executable (overrides config)"
    )

    parser.add_argument(
        "--parameter-pattern",
        type=str,
        default="auto",
        help="Pattern for extracting parameter from filename (default: auto-detect)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running simulations",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser


def create_parameter_extractor(pattern: str):
    """Create a parameter extraction function based on pattern."""

    if pattern == "auto":
        # Default pattern: extract number before 'u.txt'
        def extractor(fname):
            try:
                # Handle patterns like "waveform_80u.txt" -> 80
                parts = fname.replace(".txt", "").split("_")
                for part in reversed(parts):
                    if part.endswith("u"):
                        return int(part[:-1])
                # Fallback: try to extract any number
                import re

                numbers = re.findall(r"-?\d+", fname)
                return int(numbers[-1]) if numbers else 0
            except (ValueError, IndexError):
                return 0

        return extractor

    elif pattern == "simple":
        # Simple pattern: just extract the last number
        def extractor(fname):
            import re

            numbers = re.findall(r"-?\d+", fname)
            return int(numbers[-1]) if numbers else 0

        return extractor

    else:
        # Custom pattern (would need more sophisticated implementation)
        raise NotImplementedError(f"Custom pattern '{pattern}' not implemented")


def run_simulation_sweep(
    config,
    template_path: Path,
    wave_dir: Path,
    output_path: Optional[Path],
    ltspice_path: Optional[str],
    parameter_pattern: str,
    dry_run: bool = False,
):
    """Run the simulation parameter sweep."""

    logger = logging.getLogger(__name__)

    # Setup LTspice runner
    ltspice_exe = ltspice_path or config.simulation.ltspice_path
    runner = LTspiceRunner(ltspice_exe)

    # Setup automator
    automator = SimulationAutomator(runner)

    # Create parameter extractor
    param_extractor = create_parameter_extractor(parameter_pattern)

    # Check inputs
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    if not wave_dir.exists():
        raise FileNotFoundError(f"Waveform directory not found: {wave_dir}")

    # Find PWL files
    pwl_files = list(wave_dir.glob("*.txt"))
    if not pwl_files:
        raise ValueError(f"No PWL files found in {wave_dir}")

    logger.info(f"Found {len(pwl_files)} PWL files to process")

    if dry_run:
        logger.info("DRY RUN - No simulations will be executed")
        for fname in sorted(pwl_files):
            param_value = param_extractor(fname.name)
            logger.info(f"Would process: {fname.name} -> parameter = {param_value}")
        return None

    # Set default output path if not provided
    if output_path is None:
        output_path = get_results_dir() / "parameter_sweep_results.csv"

    # Run the sweep
    logger.info(f"Starting parameter sweep with template: {template_path}")
    logger.info(f"Processing waveforms from: {wave_dir}")
    logger.info(f"Results will be saved to: {output_path}")

    results_df = automator.run_parameter_sweep(
        template_netlist=template_path,
        wave_dir=wave_dir,
        output_csv=output_path,
        parameter_extractor=param_extractor,
    )

    logger.info(
        f"Sweep completed successfully! Processed {len(results_df)} simulations"
    )

    # Print summary
    if len(results_df) > 0:
        print("\nSweep Results Summary:")
        print(
            f"Parameter range: {results_df['Parameter'].min()} to {results_df['Parameter'].max()}"
        )
        print(
            f"Persistent current range: {results_df['Persistent_Current_uA'].min():.2f} to {results_df['Persistent_Current_uA'].max():.2f} µA"
        )
        print(f"Results saved to: {output_path}")

    return results_df


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        if args.config:
            config = ConfigManager.load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = get_default_config()
            logger.info("Using default configuration")

        # Convert paths
        template_path = Path(args.template)
        wave_dir = Path(args.wave_dir)
        output_path = Path(args.output) if args.output else None

        # Run simulation sweep
        results = run_simulation_sweep(
            config=config,
            template_path=template_path,
            wave_dir=wave_dir,
            output_path=output_path,
            ltspice_path=args.ltspice_path,
            parameter_pattern=args.parameter_pattern,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            logger.info("Automation completed successfully!")

    except Exception as e:
        logger.error(f"Automation failed: {e}")
        raise


if __name__ == "__main__":
    main()
