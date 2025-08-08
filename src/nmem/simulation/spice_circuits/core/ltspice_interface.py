"""LTspice simulation interface and automation."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import ltspice
import pandas as pd
from ..utils.constants import DEFAULT_LTSPICE_PATH, SPICE_NETLIST_EXT, SPICE_RAW_EXT
from ..utils.file_io import ensure_directory_exists


class LTspiceRunner:
    """Class for managing LTspice simulations."""

    def __init__(self, ltspice_path: str = DEFAULT_LTSPICE_PATH):
        """Initialize LTspice runner.

        Args:
            ltspice_path: Path to LTspice executable
        """
        self.ltspice_path = ltspice_path
        self.logger = logging.getLogger(__name__)

    def _convert_path_for_ltspice(self, linux_path: Union[str, Path]) -> str:
        """Convert Linux path to Windows path format for LTspice.

        Args:
            linux_path: Linux file path

        Returns:
            Windows-compatible path for LTspice
        """
        # Convert to absolute path first
        path_obj = Path(linux_path).resolve()
        path_str = str(path_obj)

        # Convert /mnt/c/ paths to C:\ format
        if path_str.startswith("/mnt/c/"):
            return path_str.replace("/mnt/c/", "C:\\").replace("/", "\\")
        elif path_str.startswith("/mnt/"):
            # Handle other drive letters
            parts = path_str.split("/")
            if len(parts) > 2:
                drive = parts[2].upper()
                remainder = "/".join(parts[3:])
                return f"{drive}:\\{remainder}".replace("/", "\\")

        # For paths not on mounted drives, use WSL path format
        return f"\\\\wsl.localhost\\Ubuntu-22.04\\{path_str[1:]}".replace("/", "\\")

    def generate_netlist(
        self,
        template_path: Union[str, Path],
        chan_pwl_path: Union[str, Path],
        enab_pwl_path: Union[str, Path],
        output_path: Union[str, Path],
        substitutions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Generate a netlist from template with PWL file substitutions.

        Args:
            template_path: Path to netlist template
            chan_pwl_path: Path to channel PWL waveform file
            enab_pwl_path: Path to enable PWL waveform file
            output_path: Path for generated netlist
            substitutions: Additional string substitutions to make
        """
        template_path = Path(template_path)
        output_path = Path(output_path)

        with open(template_path, "r") as f:
            netlist = f.read()

        # Convert paths to Windows format for LTspice
        chan_pwl_win_path = self._convert_path_for_ltspice(chan_pwl_path)
        enab_pwl_win_path = self._convert_path_for_ltspice(enab_pwl_path)

        # Default substitutions for PWL files
        netlist = netlist.replace("CHANPWL", f'"{chan_pwl_win_path}"')
        netlist = netlist.replace("ENABPWL", f'"{enab_pwl_win_path}"')

        # Fix library path - make it relative to the template directory
        circuit_files_dir = template_path.parent
        lib_path = circuit_files_dir / "hTron_behavioral.lib"
        lib_win_path = self._convert_path_for_ltspice(lib_path)
        netlist = netlist.replace("hTron_behavioral.lib", f'"{lib_win_path}"')

        # Apply additional substitutions if provided
        if substitutions:
            for old, new in substitutions.items():
                netlist = netlist.replace(old, new)

        # Ensure output directory exists
        ensure_directory_exists(output_path.parent)

        with open(output_path, "w") as f:
            f.write(netlist)

        self.logger.info(f"Generated netlist: {output_path}")

    def generate_netlist_single_pwl(
        self,
        template_path: Union[str, Path],
        pwl_path: Union[str, Path],
        output_path: Union[str, Path],
        substitutions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Generate a netlist from template with single PWL file substitution.

        This method is for backward compatibility with templates that use CHANPWL only.

        Args:
            template_path: Path to netlist template
            pwl_path: Path to PWL waveform file
            output_path: Path for generated netlist
            substitutions: Additional string substitutions to make
        """
        template_path = Path(template_path)
        output_path = Path(output_path)

        with open(template_path, "r") as f:
            netlist = f.read()

        # Convert path to Windows format for LTspice
        pwl_win_path = self._convert_path_for_ltspice(pwl_path)

        # Default substitution for PWL file
        netlist = netlist.replace("CHANPWL", f'"{pwl_win_path}"')

        # Fix library path - make it relative to the template directory
        circuit_files_dir = template_path.parent
        lib_path = circuit_files_dir / "hTron_behavioral.lib"
        lib_win_path = self._convert_path_for_ltspice(lib_path)
        netlist = netlist.replace("hTron_behavioral.lib", f'"{lib_win_path}"')

        # Apply additional substitutions if provided
        if substitutions:
            for old, new in substitutions.items():
                netlist = netlist.replace(old, new)

        # Ensure output directory exists
        ensure_directory_exists(output_path.parent)

        with open(output_path, "w") as f:
            f.write(netlist)

        self.logger.info(f"Generated netlist: {output_path}")

    def run_simulation(self, netlist_path: Union[str, Path]) -> None:
        """Run LTspice simulation in headless mode.

        Args:
            netlist_path: Path to netlist file
        """
        # Convert path for LTspice if needed
        ltspice_netlist_path = self._convert_path_for_ltspice(netlist_path)

        try:
            # Use -b for batch mode and -Run for headless operation
            cmd = [self.ltspice_path, "-b", "-Run", ltspice_netlist_path]
            self.logger.info(f"Running LTspice command: {' '.join(cmd)}")

            # Run with no window (headless)
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0
                ),
            )

            self.logger.info(f"LTspice simulation completed for: {netlist_path}")
            if result.stdout:
                self.logger.debug(f"LTspice output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"LTspice simulation failed: {e}")
            if e.stderr:
                self.logger.error(f"LTspice error output: {e.stderr}")
            raise
        except FileNotFoundError:
            self.logger.error(f"LTspice executable not found at: {self.ltspice_path}")
            raise

    def parse_raw_file(self, raw_path: Union[str, Path]) -> ltspice.Ltspice:
        """Parse LTspice .raw output file.

        Args:
            raw_path: Path to .raw file

        Returns:
            Parsed LTspice data object
        """
        ltsp = ltspice.Ltspice(str(raw_path))
        ltsp.parse()
        self.logger.info(f"Parsed raw file: {raw_path}")
        return ltsp

    def extract_persistent_current(
        self, raw_path: Union[str, Path], time_range: tuple = (3.9e-6, 4.1e-6)
    ) -> float:
        """Extract persistent current from simulation results.

        Args:
            raw_path: Path to .raw file
            time_range: Time range for averaging (start, end)

        Returns:
            Persistent current in µA
        """
        ltsp = self.parse_raw_file(raw_path)
        time = ltsp.get_time()
        ir = ltsp.get_data("Ix(HR:drain)")

        mask = (time >= time_range[0]) & (time <= time_range[1])
        persistent_current = float(np.mean(ir[mask]) * 1e6)  # µA

        self.logger.info(f"Extracted persistent current: {persistent_current:.2f} µA")
        return persistent_current


class SimulationAutomator:
    """Class for automating simulation sweeps."""

    def __init__(self, ltspice_runner: LTspiceRunner):
        """Initialize automation with LTspice runner.

        Args:
            ltspice_runner: Configured LTspice runner instance
        """
        self.runner = ltspice_runner
        self.logger = logging.getLogger(__name__)

    def run_parameter_sweep(
        self,
        template_netlist: Union[str, Path],
        wave_dir: Union[str, Path],
        output_csv: Union[str, Path],
        parameter_extractor: callable = None,
    ) -> pd.DataFrame:
        """Run a parameter sweep simulation.

        Args:
            template_netlist: Path to netlist template
            wave_dir: Directory containing PWL waveform files
            output_csv: Path for results CSV file
            parameter_extractor: Function to extract parameter from filename

        Returns:
            DataFrame with simulation results
        """
        wave_dir = Path(wave_dir)
        output_data = []

        # Default parameter extractor if none provided
        if parameter_extractor is None:
            parameter_extractor = lambda fname: int(
                fname.split("_")[-1].replace("u.txt", "")
            )

        for fname in sorted(wave_dir.glob("*.txt")):
            try:
                # Extract parameter value from filename
                param_value = parameter_extractor(fname.name)

                # Generate temporary netlist
                temp_netlist = wave_dir / f"temp_{param_value:+05g}u.cir"
                self.runner.generate_netlist_single_pwl(
                    template_netlist, fname, temp_netlist
                )

                # Run simulation
                self.runner.run_simulation(temp_netlist)

                # Parse results
                raw_path = temp_netlist.with_suffix(".raw")
                persistent_current = self.runner.extract_persistent_current(raw_path)

                output_data.append(
                    {
                        "Parameter": param_value,
                        "Persistent_Current_uA": persistent_current,
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing {fname}: {e}")
                continue

        # Save results
        df = pd.DataFrame(output_data)
        if output_csv:
            df.to_csv(output_csv, index=False)
            self.logger.info(f"Results saved to {output_csv}")

        return df

    def run_waveform_simulation(
        self,
        template_netlist: Union[str, Path],
        chan_pwl_path: Union[str, Path],
        enab_pwl_path: Union[str, Path],
        output_netlist: Union[str, Path],
        extract_results: bool = True,
    ) -> Optional[float]:
        """Run a simulation with specific channel and enable waveform files.

        Args:
            template_netlist: Path to netlist template
            chan_pwl_path: Path to channel PWL file
            enab_pwl_path: Path to enable PWL file
            output_netlist: Path for generated netlist
            extract_results: Whether to extract and return persistent current

        Returns:
            Persistent current in µA if extract_results=True, otherwise None
        """
        try:
            # Generate netlist with both PWL files
            self.runner.generate_netlist(
                template_netlist, chan_pwl_path, enab_pwl_path, output_netlist
            )

            # Run simulation
            self.runner.run_simulation(output_netlist)

            # Extract results if requested
            if extract_results:
                raw_path = Path(output_netlist).with_suffix(".raw")
                return self.runner.extract_persistent_current(raw_path)

        except Exception as e:
            self.logger.error(f"Error running waveform simulation: {e}")
            raise

        return None


def load_ltspice_data(raw_dir: Union[str, Path]) -> Dict[str, ltspice.Ltspice]:
    """Load multiple LTspice data files from a directory.

    Args:
        raw_dir: Directory containing .raw files

    Returns:
        Dictionary mapping filenames to parsed LTspice objects
    """
    raw_dir = Path(raw_dir)
    data_dict = {}

    for raw_file in raw_dir.glob("*.raw"):
        try:
            ltsp = ltspice.Ltspice(str(raw_file))
            ltsp.parse()
            data_dict[raw_file.stem] = ltsp
        except Exception as e:
            logging.warning(f"Failed to load {raw_file}: {e}")

    return data_dict
