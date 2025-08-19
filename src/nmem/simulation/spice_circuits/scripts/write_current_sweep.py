#!/usr/bin/env python3
"""
Write Current Sweep Analysis

This script tracks the persistent current stored in the memory cell as a function
of the input write current amplitude. It uses the existing waveform patterns from
get_default_patterns and sweeps the write current amplitude while keeping other
parameters constant.

Usage:
    python write_current_sweep.py --write-current-range 50 100 5 --output results/write_current_sweep.csv
    python write_current_sweep.py --config config.yaml --write-current-range 20 120 10
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from datetime import datetime

try:
    from ..core.ltspice_interface import LTspiceRunner, SimulationAutomator
    from ..waveform.generators import WaveformGenerator
    from ..utils.file_io import (
        get_waveforms_dir,
        get_results_dir,
        get_circuit_files_dir,
        ensure_directory_exists,
        save_pwl_file,
    )
    from ..utils.constants import *
except ImportError:
    # Standalone execution - add project root to path
    import sys

    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from nmem.simulation.spice_circuits.core.ltspice_interface import (
        LTspiceRunner,
        SimulationAutomator,
    )
    from nmem.simulation.spice_circuits.waveform.generators import WaveformGenerator
    from nmem.simulation.spice_circuits.utils.file_io import (
        get_waveforms_dir,
        get_results_dir,
        get_circuit_files_dir,
        ensure_directory_exists,
        save_pwl_file,
    )
    from nmem.simulation.spice_circuits.utils.constants import *


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


class WriteCurrentSweepAnalyzer:
    """Class for running write current sweep analysis."""

    def __init__(
        self,
        ltspice_path: str = DEFAULT_LTSPICE_PATH,
        base_config: Optional[Dict] = None,
    ):
        """Initialize the analyzer.

        Args:
            ltspice_path: Path to LTspice executable
            base_config: Base configuration for waveform generation
        """
        self.logger = logging.getLogger(__name__)
        self.ltspice_runner = LTspiceRunner(ltspice_path)
        self.automator = SimulationAutomator(self.ltspice_runner)

        # Default waveform configuration
        self.base_config = base_config or {
            "cycle_time": DEFAULT_CYCLE_TIME,
            "pulse_sigma": DEFAULT_PULSE_SIGMA,
            "hold_width_write": DEFAULT_HOLD_WIDTH_WRITE,
            "hold_width_read": DEFAULT_HOLD_WIDTH_READ,
            "hold_width_clear": DEFAULT_HOLD_WIDTH_CLEAR,
            "read_amplitude": DEFAULT_READ_AMPLITUDE,
            "enab_write_amplitude": DEFAULT_ENAB_WRITE_AMPLITUDE,
            "enab_read_amplitude": DEFAULT_ENAB_READ_AMPLITUDE,
            "clear_amplitude": DEFAULT_CLEAR_AMPLITUDE,
            "dt": DEFAULT_DT,
            "seed": 42,
        }

    def generate_waveforms_for_write_current(
        self, write_current_ua: float, output_dir: Path
    ) -> Tuple[Path, Path]:
        """Generate waveforms for a specific write current amplitude.

        Args:
            write_current_ua: Write current amplitude in microamps
            output_dir: Directory to save waveform files

        Returns:
            Tuple of (channel_waveform_path, enable_waveform_path)
        """
        ensure_directory_exists(output_dir)

        # Create waveform generator with specific write amplitude
        config = self.base_config.copy()
        config["write_amplitude"] = write_current_ua * 1e-6  # Convert µA to A

        generator = WaveformGenerator(
            **{k: v for k, v in config.items() if k != "seed"}
        )

        # Generate waveforms using default patterns
        t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
            generator.generate_memory_protocol_sequence(
                patterns=generator.get_default_patterns(), seed=config["seed"]
            )
        )

        # Save waveform files
        chan_file = output_dir / f"chan_write_{write_current_ua:06.1f}uA.txt"
        enab_file = output_dir / f"enab_write_{write_current_ua:06.1f}uA.txt"

        save_pwl_file(chan_file, t_chan, i_chan)
        save_pwl_file(enab_file, t_enab, i_enab)

        self.logger.debug(
            f"Generated waveforms for {write_current_ua}µA: {chan_file.name}, {enab_file.name}"
        )

        return chan_file, enab_file

    def run_single_simulation(
        self,
        write_current_ua: float,
        chan_file: Path,
        enab_file: Path,
        template_netlist: Path,
        output_dir: Path,
        simulation_config: Optional[Dict] = None,
    ) -> Optional[float]:
        """Run a single simulation for a given write current.

        Args:
            write_current_ua: Write current amplitude in microamps
            chan_file: Path to channel waveform file
            enab_file: Path to enable waveform file
            template_netlist: Path to circuit template
            output_dir: Directory for simulation outputs
            simulation_config: Additional simulation parameters

        Returns:
            Persistent current in microamps, or None if simulation failed
        """
        try:
            # Create unique output netlist
            output_netlist = (
                output_dir / f"simulation_write_{write_current_ua:06.1f}uA.cir"
            )

            # Prepare simulation parameters
            substitutions = {
                "{start_time}": "0",
                "{stop_time}": "4.5e-6",
                "{start_save}": "0",
                "{time_step}": "1e-9",
            }

            if simulation_config:
                substitutions.update(simulation_config)

            # Generate netlist
            self.ltspice_runner.generate_netlist(
                template_path=template_netlist,
                chan_pwl_path=chan_file,
                enab_pwl_path=enab_file,
                output_path=output_netlist,
                substitutions=substitutions,
            )

            # Run simulation
            self.ltspice_runner.run_simulation(output_netlist)

            # Extract persistent current
            raw_file = output_netlist.with_suffix(".raw")
            if raw_file.exists():
                persistent_current = self.ltspice_runner.extract_persistent_current(
                    raw_file, time_range=(3.9e-6, 4.1e-6)
                )
                return persistent_current
            else:
                self.logger.error(f"Raw file not found: {raw_file}")
                return None

        except Exception as e:
            self.logger.error(f"Simulation failed for {write_current_ua}µA: {e}")
            return None

    def run_write_current_sweep(
        self,
        write_current_range: Tuple[float, float, float],
        template_netlist: Optional[Path] = None,
        output_csv: Optional[Path] = None,
        simulation_config: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Run a complete write current sweep.

        Args:
            write_current_range: Tuple of (start, stop, step) in microamps
            template_netlist: Path to circuit template (uses default if None)
            output_csv: Path to save results CSV (auto-generated if None)
            simulation_config: Additional simulation parameters

        Returns:
            DataFrame with sweep results
        """
        start_ua, stop_ua, step_ua = write_current_range
        write_currents = np.arange(start_ua, stop_ua + step_ua / 2, step_ua)

        self.logger.info(
            f"Starting write current sweep: {start_ua}µA to {stop_ua}µA, step {step_ua}µA"
        )
        self.logger.info(f"Total simulations: {len(write_currents)}")

        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = get_results_dir() / f"write_current_sweep_{timestamp}"
        waveforms_dir = sweep_dir / "waveforms"
        simulations_dir = sweep_dir / "simulations"

        ensure_directory_exists(waveforms_dir)
        ensure_directory_exists(simulations_dir)

        # Default template if not provided
        if template_netlist is None:
            template_netlist = get_circuit_files_dir() / "default_circuit_template.cir"

        if not template_netlist.exists():
            raise FileNotFoundError(f"Template netlist not found: {template_netlist}")

        # Results storage
        results = []

        for i, write_current_ua in enumerate(write_currents):
            self.logger.info(
                f"Processing {i+1}/{len(write_currents)}: {write_current_ua}µA"
            )

            try:
                # Generate waveforms
                chan_file, enab_file = self.generate_waveforms_for_write_current(
                    write_current_ua, waveforms_dir
                )

                # Run simulation
                persistent_current = self.run_single_simulation(
                    write_current_ua,
                    chan_file,
                    enab_file,
                    template_netlist,
                    simulations_dir,
                    simulation_config,
                )

                if persistent_current is not None:
                    results.append(
                        {
                            "Write_Current_uA": write_current_ua,
                            "Persistent_Current_uA": persistent_current,
                            "Efficiency_Percent": (
                                (persistent_current / write_current_ua) * 100
                                if write_current_ua != 0
                                else 0
                            ),
                            "Chan_Waveform": str(chan_file.relative_to(sweep_dir)),
                            "Enab_Waveform": str(enab_file.relative_to(sweep_dir)),
                            "Timestamp": datetime.now().isoformat(),
                        }
                    )

                    self.logger.info(
                        f"  ✅ Write: {write_current_ua}µA → Persistent: {persistent_current:.2f}µA"
                    )
                else:
                    self.logger.warning(
                        f"  ❌ Simulation failed for {write_current_ua}µA"
                    )

            except Exception as e:
                self.logger.error(f"  ❌ Error processing {write_current_ua}µA: {e}")
                continue

        # Create results DataFrame
        df = pd.DataFrame(results)

        if len(df) > 0:
            # Save results
            if output_csv is None:
                output_csv = sweep_dir / "write_current_sweep_results.csv"
            else:
                output_csv = Path(output_csv)
                ensure_directory_exists(output_csv.parent)

            df.to_csv(output_csv, index=False)

            # Save configuration
            config_file = sweep_dir / "sweep_config.yaml"
            config_data = {
                "write_current_range": write_current_range,
                "base_config": self.base_config,
                "template_netlist": str(template_netlist),
                "simulation_config": simulation_config or {},
                "timestamp": timestamp,
                "total_simulations": len(write_currents),
                "successful_simulations": len(df),
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            # Print summary
            self.logger.info(f"\n✅ Sweep completed successfully!")
            self.logger.info(
                f"   Successful simulations: {len(df)}/{len(write_currents)}"
            )
            self.logger.info(
                f"   Write current range: {df['Write_Current_uA'].min():.1f} - {df['Write_Current_uA'].max():.1f} µA"
            )
            self.logger.info(
                f"   Persistent current range: {df['Persistent_Current_uA'].min():.2f} - {df['Persistent_Current_uA'].max():.2f} µA"
            )
            if len(df) > 0:
                max_efficiency_idx = df["Efficiency_Percent"].idxmax()
                optimal_write = df.loc[max_efficiency_idx, "Write_Current_uA"]
                max_efficiency = df.loc[max_efficiency_idx, "Efficiency_Percent"]
                self.logger.info(
                    f"   Maximum efficiency: {max_efficiency:.1f}% at {optimal_write:.1f}µA write current"
                )
            self.logger.info(f"   Results saved to: {output_csv}")
            self.logger.info(f"   Configuration saved to: {config_file}")

        else:
            self.logger.error("❌ No successful simulations completed!")

        return df


def load_config_from_file(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_sample_config() -> Dict:
    """Create a sample configuration for the sweep."""
    return {
        "waveform_config": {
            "cycle_time": 1e-6,
            "pulse_sigma": 35e-9,
            "hold_width_write": 120e-9,
            "hold_width_read": 300e-9,
            "hold_width_clear": 5e-9,
            "read_amplitude": 725e-6,
            "enab_write_amplitude": 465e-6,
            "enab_read_amplitude": 300e-6,
            "clear_amplitude": 700e-6,
            "dt": 0.1e-9,
            "seed": 42,
        },
        "simulation_config": {
            "{start_time}": "0",
            "{stop_time}": "4.5e-6",
            "{start_save}": "0",
            "{time_step}": "1e-9",
        },
        "ltspice_path": "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
    }


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run write current sweep analysis for memory cells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic sweep from 50µA to 100µA in 5µA steps
    python write_current_sweep.py --write-current-range 50 100 5
    
    # Extended sweep with custom output
    python write_current_sweep.py --write-current-range 20 120 10 --output results/my_sweep.csv
    
    # Use custom config file
    python write_current_sweep.py --config sweep_config.yaml --write-current-range 40 80 5
    
    # Create sample config file
    python write_current_sweep.py --create-config sample_config.yaml
        """,
    )

    parser.add_argument(
        "--write-current-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help="Write current range: start stop step (all in µA)",
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Configuration file path (YAML)"
    )

    parser.add_argument(
        "--template",
        type=str,
        help="Path to SPICE netlist template (default: use built-in template)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file path (auto-generated if not specified)",
    )

    parser.add_argument(
        "--ltspice-path", type=str, help="Path to LTspice executable (overrides config)"
    )

    parser.add_argument(
        "--create-config",
        type=str,
        metavar="CONFIG_FILE",
        help="Create a sample configuration file",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file", type=str, help="Log file path (logs to console if not specified)"
    )

    return parser


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Create sample config if requested
        if args.create_config:
            config = create_sample_config()
            with open(args.create_config, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"✅ Created sample configuration: {args.create_config}")
            return

        # Check required arguments
        if not args.write_current_range:
            logger.error(
                "❌ Write current range is required (--write-current-range START STOP STEP)"
            )
            return

        # Load configuration
        if args.config:
            config = load_config_from_file(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = create_sample_config()
            logger.info("Using default configuration")

        # Setup analyzer
        ltspice_path = args.ltspice_path or config.get(
            "ltspice_path", DEFAULT_LTSPICE_PATH
        )
        analyzer = WriteCurrentSweepAnalyzer(
            ltspice_path=ltspice_path, base_config=config.get("waveform_config", {})
        )

        # Setup paths
        template_path = Path(args.template) if args.template else None
        output_path = Path(args.output) if args.output else None

        # Run sweep
        logger.info("🚀 Starting write current sweep analysis...")
        results_df = analyzer.run_write_current_sweep(
            write_current_range=tuple(args.write_current_range),
            template_netlist=template_path,
            output_csv=output_path,
            simulation_config=config.get("simulation_config", {}),
        )

        if len(results_df) > 0:
            logger.info("🎉 Analysis completed successfully!")
        else:
            logger.error("❌ Analysis failed - no successful simulations")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
