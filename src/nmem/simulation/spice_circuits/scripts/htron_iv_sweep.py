#!/usr/bin/env python3
"""
hTron IV Sweep Analysis

This script performs current-voltage (IV) measurements on a single hTron device
by sweeping bias current and optionally heater current. It generates IV curves
to characterize the switching and retrapping behavior.

Usage:
    python htron_iv_sweep.py --bias-range 0 500 50 --heater-current 0
    python htron_iv_sweep.py --bias-range 100 400 25 --heater-range 0 500 100
    python htron_iv_sweep.py --config iv_config.yaml
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
    from ..utils.file_io import (
        get_results_dir,
        get_circuit_files_dir,
        ensure_directory_exists,
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
    from nmem.simulation.spice_circuits.utils.file_io import (
        get_results_dir,
        get_circuit_files_dir,
        ensure_directory_exists,
    )


class HTronIVSweep:
    """Class to perform IV sweep measurements on hTron devices."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the IV sweep analyzer."""
        self.config = self._load_config(config_path)
        self.ltspice_runner = LTspiceRunner(self.config.get("ltspice_path"))
        self.results_dir = None
        self.logger = self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "ltspice_path": "/mnt/c/Users/omedeiro/AppData/Local/Programs/ADI/LTspice/LTspice.exe",
            "circuit_template": "htron_iv_template.cir",
            "bias_current_range": [0, 500e-6, 50e-6],  # Start, stop, step in Amps
            "heater_current_range": [0, 500e-6, 100e-6],  # Start, stop, step in Amps
            "simulation_time": 400e-6,  # Total simulation time in seconds
            "device_parameters": {
                "chan_width": "100n",
                "chan_length": "1u",
                "heater_resistance": "300",
                "critical_temp": "12.5",
                "substrate_temp": "1.3",
                "sheet_resistance": "78",
                "Jsw_tilde": "250G",
                "Isupp_tilde": "540u",
                "Jchanr": "99G",
            },
            "output_signals": [
                "I(X§U1:Lc)",
                "V(out)",
                "V(tempCh)",
                "I(I2)",
                "V(Meas_Isw)",
                "V(Meas_Ihs)",
            ],
        }

        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
            # Merge user config with defaults
            default_config.update(user_config)

            # Convert new format to old format if needed
            if "measurements" in user_config:
                measurements = user_config["measurements"]

                # Convert bias current configuration
                if "bias_currents" in measurements:
                    bias_config = measurements["bias_currents"]
                    start = float(bias_config.get("start", 0))
                    stop = float(bias_config.get("stop", 500e-6))
                    steps = int(bias_config.get("steps", 11))
                    step_size = (stop - start) / (steps - 1) if steps > 1 else 0
                    default_config["bias_current_range"] = [start, stop, step_size]

                # Convert heater current configuration
                if "heater_currents" in measurements:
                    heater_config = measurements["heater_currents"]
                    start = float(heater_config.get("start", 0))
                    stop = float(heater_config.get("stop", 500e-6))
                    steps = int(heater_config.get("steps", 6))
                    step_size = (stop - start) / (steps - 1) if steps > 1 else 0
                    default_config["heater_current_range"] = [start, stop, step_size]

        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("HTronIVSweep")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def setup_output_directory(self, base_name: str = "htron_iv_sweep") -> Path:
        """Create timestamped output directory for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = get_results_dir() / f"{base_name}_{timestamp}"
        ensure_directory_exists(self.results_dir)

        # Create subdirectories
        (self.results_dir / "simulations").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)

        self.logger.info(f"📁 Results directory: {self.results_dir}")
        return self.results_dir

    def generate_bias_current_values(self) -> np.ndarray:
        """Generate array of bias current values to sweep."""
        start, stop, step = self.config["bias_current_range"]
        if start == stop or step == 0:
            return np.array([start])
        return np.arange(start, stop + step, step)

    def generate_heater_current_values(self) -> np.ndarray:
        """Generate array of heater current values to sweep."""
        start, stop, step = self.config["heater_current_range"]
        if start == stop or step == 0:
            return np.array([start])
        return np.arange(start, stop + step, step)

    def run_single_iv_simulation(
        self, bias_current: float, heater_current: float
    ) -> Tuple[bool, Optional[Path]]:
        """
        Run a single IV simulation for given bias and heater currents.

        Args:
            bias_current: Channel bias current in Amps
            heater_current: Heater current in Amps

        Returns:
            Tuple of (success, raw_file_path)
        """
        try:
            # Create circuit file with parameters
            circuit_template_path = (
                get_circuit_files_dir() / self.config["circuit_template"]
            )

            # Read template
            with open(circuit_template_path, "r") as f:
                circuit_content = f.read()

            # Replace only the bias current parameters (simplified approach)
            modified_circuit = circuit_content.replace(
                ".param I_bias=300u I_heater=500u",
                f".param I_bias={bias_current:.2e} I_heater={heater_current:.2e}",
            )

            # Fix library path like the write sweep does
            circuit_files_dir = circuit_template_path.parent
            lib_path = circuit_files_dir / "hTron_behavioral.lib"
            lib_win_path = self.ltspice_runner._convert_path_for_ltspice(lib_path)
            modified_circuit = modified_circuit.replace(
                "hTron_behavioral.lib", f'"{lib_win_path}"'
            )

            # Create simulation-specific circuit file
            sim_name = (
                f"iv_bias_{bias_current*1e6:.0f}uA_heater_{heater_current*1e6:.0f}uA"
            )
            circuit_path = self.results_dir / "simulations" / f"{sim_name}.cir"

            with open(circuit_path, "w") as f:
                f.write(modified_circuit)

            # Run simulation
            self.logger.info(
                f"🔄 Running: Bias={bias_current*1e6:.0f}µA, Heater={heater_current*1e6:.0f}µA"
            )

            raw_file = self.ltspice_runner.run_simulation(circuit_path)
            if raw_file and raw_file.exists():
                return True, raw_file
            else:
                self.logger.error(f"❌ Simulation failed for {sim_name}")
                return False, None

        except Exception as e:
            self.logger.error(f"❌ Error in simulation {sim_name}: {e}")
            return False, None

    def extract_iv_data(self, raw_file: Path) -> Dict[str, np.ndarray]:
        """
        Extract IV data from simulation results.

        Args:
            raw_file: Path to LTspice .raw file

        Returns:
            Dictionary with time, current, voltage, and other signals
        """
        try:
            import ltspice

            ltsp = ltspice.Ltspice(str(raw_file))
            ltsp.parse()

            # Extract time and signals
            time = ltsp.get_time()

            data = {"time": time}

            for signal in self.config["output_signals"]:
                try:
                    signal_data = ltsp.get_data(signal)
                    data[signal] = signal_data
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not extract signal {signal}: {e}")

            return data

        except Exception as e:
            self.logger.error(f"❌ Error extracting data from {raw_file}: {e}")
            return {}

    def run_iv_sweep(self) -> pd.DataFrame:
        """
        Run complete IV sweep over bias and heater current ranges.

        Returns:
            DataFrame with IV sweep results
        """
        self.logger.info("🚀 Starting hTron IV sweep analysis")

        # Setup output directory
        self.setup_output_directory()

        # Generate current ranges
        bias_currents = self.generate_bias_current_values()
        heater_currents = self.generate_heater_current_values()

        self.logger.info(
            f"📊 Bias current range: {bias_currents[0]*1e6:.0f} to {bias_currents[-1]*1e6:.0f} µA ({len(bias_currents)} points)"
        )
        self.logger.info(
            f"📊 Heater current range: {heater_currents[0]*1e6:.0f} to {heater_currents[-1]*1e6:.0f} µA ({len(heater_currents)} points)"
        )

        # Collect results
        results = []
        total_sims = len(bias_currents) * len(heater_currents)
        sim_count = 0

        for heater_current in heater_currents:
            for bias_current in bias_currents:
                sim_count += 1
                self.logger.info(f"📈 Progress: {sim_count}/{total_sims}")

                # Run simulation
                success, raw_file = self.run_single_iv_simulation(
                    bias_current, heater_current
                )

                if success and raw_file:
                    # Extract data
                    iv_data = self.extract_iv_data(raw_file)

                    if iv_data:
                        # Calculate key metrics from the IV data
                        time = iv_data.get("time", np.array([]))
                        voltage = iv_data.get("V(out)", np.array([]))
                        current = iv_data.get("I(I2)", np.array([]))

                        if len(voltage) > 0 and len(current) > 0:
                            # Find key points in the IV sweep
                            max_voltage = (
                                np.max(np.abs(voltage)) if len(voltage) > 0 else 0
                            )
                            max_current = (
                                np.max(np.abs(current)) if len(current) > 0 else 0
                            )

                            # Store result
                            result = {
                                "Bias_Current_uA": bias_current * 1e6,
                                "Heater_Current_uA": heater_current * 1e6,
                                "Max_Voltage_mV": max_voltage * 1e3,
                                "Max_Current_uA": max_current * 1e6,
                                "Raw_File": str(raw_file),
                                "Success": True,
                            }

                            # Add additional extracted signals
                            for signal, data in iv_data.items():
                                if signal not in ["time"] and len(data) > 0:
                                    result[
                                        f"Max_{signal.replace('(', '').replace(')', '').replace(':', '_')}"
                                    ] = np.max(np.abs(data))

                            results.append(result)
                        else:
                            self.logger.warning(f"⚠️  No voltage/current data extracted")
                    else:
                        self.logger.warning(f"⚠️  No data extracted from simulation")
                else:
                    # Record failed simulation
                    results.append(
                        {
                            "Bias_Current_uA": bias_current * 1e6,
                            "Heater_Current_uA": heater_current * 1e6,
                            "Max_Voltage_mV": np.nan,
                            "Max_Current_uA": np.nan,
                            "Raw_File": "",
                            "Success": False,
                        }
                    )

        # Convert to DataFrame
        df_results = pd.DataFrame(results)

        # Save results
        results_file = self.results_dir / "htron_iv_sweep_results.csv"
        df_results.to_csv(results_file, index=False)
        self.logger.info(f"💾 Results saved to: {results_file}")

        # Save configuration
        config_file = self.results_dir / "sweep_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.logger.info(
            f"🎉 IV sweep complete! Processed {len(df_results)} simulations"
        )
        return df_results


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Run hTron IV sweep analysis")

    parser.add_argument(
        "--config", "-c", type=Path, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--bias-range",
        "-b",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help="Bias current range in µA: start stop step",
    )

    parser.add_argument(
        "--heater-range",
        nargs=3,
        type=float,
        metavar=("START", "STOP", "STEP"),
        help="Heater current range in µA: start stop step",
    )

    parser.add_argument(
        "--heater-current",
        type=float,
        help="Single heater current value in µA (overrides heater-range)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (will be created with timestamp)",
    )

    parser.add_argument("--ltspice-path", type=Path, help="Path to LTspice executable")

    args = parser.parse_args()

    # Create analyzer
    analyzer = HTronIVSweep(args.config)

    # Override config with command line arguments
    if args.bias_range:
        analyzer.config["bias_current_range"] = [x * 1e-6 for x in args.bias_range]

    if args.heater_current is not None:
        heater_val = args.heater_current * 1e-6
        analyzer.config["heater_current_range"] = [heater_val, heater_val, 1e-6]
    elif args.heater_range:
        analyzer.config["heater_current_range"] = [x * 1e-6 for x in args.heater_range]

    if args.ltspice_path:
        analyzer.config["ltspice_path"] = str(args.ltspice_path)

    # Run analysis
    try:
        results_df = analyzer.run_iv_sweep()
        print(f"\n🎉 IV sweep completed successfully!")
        print(f"📊 Results summary:")
        print(f"   • Total simulations: {len(results_df)}")
        print(f"   • Successful: {results_df['Success'].sum()}")
        print(f"   • Failed: {(~results_df['Success']).sum()}")
        print(f"   • Results directory: {analyzer.results_dir}")

    except Exception as e:
        print(f"❌ Error during IV sweep: {e}")
        raise


if __name__ == "__main__":
    main()
