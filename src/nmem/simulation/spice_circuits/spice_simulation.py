#!/usr/bin/env python3
"""
SPICE Simulation Main Interface

This module provides the streamlined interface for SPICE simulation automation
with three main functions:
1. generate_waveforms() - Create simulation input waveforms
2. run_simulation() - Execute SPICE simulation
3. plot_results() - Generate analysis plots

Usage:
    python spice_simulation.py --config config.yaml

    Or programmatically:
    from nmem.simulation.spice_circuits.spice_simulation import generate_waveforms, run_simulation, plot_results
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml

# Handle imports for both standalone and module usage
try:
    from .core.ltspice_interface import LTspiceRunner, SimulationAutomator
    from .core.unified_plotter import (
        UnifiedPlotter,
        PlotType,
        create_default_plotting_config,
    )
    from .waveform.generators import WaveformGenerator
    from .config.settings import get_default_config
    from .utils.file_io import (
        get_waveforms_dir,
        get_results_dir,
        get_config_dir,
        ensure_directory_exists,
        save_pwl_file,
    )
except ImportError:
    # Standalone execution - add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from nmem.simulation.spice_circuits.core.ltspice_interface import (
        LTspiceRunner,
        SimulationAutomator,
    )
    from nmem.simulation.spice_circuits.core.unified_plotter import (
        UnifiedPlotter,
        PlotType,
        create_default_plotting_config,
    )
    from nmem.simulation.spice_circuits.waveform.generators import WaveformGenerator
    from nmem.simulation.spice_circuits.config.settings import get_default_config
    from nmem.simulation.spice_circuits.utils.file_io import (
        get_waveforms_dir,
        get_results_dir,
        get_config_dir,
        ensure_directory_exists,
        save_pwl_file,
    )


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def generate_waveforms(
    config_path: Union[str, Path], output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Generate simulation input waveforms.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to save waveforms (defaults to standard waveforms dir)

    Returns:
        Dictionary mapping waveform names to file paths
    """
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)

    if output_dir is None:
        output_dir = get_waveforms_dir()
    ensure_directory_exists(output_dir)

    logger.info("🌊 Generating simulation waveforms...")

    # Extract waveform configuration
    waveform_config = config.get("waveforms", {})

    waveform_files = {}

    if "standard" in waveform_config:
        std_config = waveform_config["standard"].copy()

        # Extract seed for method call
        seed = std_config.pop("seed", 42)

        # Initialize generator with config parameters (excluding seed)
        generator = WaveformGenerator(**std_config)

        # Generate waveforms
        t_chan, i_chan, t_enab, i_enab, ops, enab_on = (
            generator.generate_memory_protocol_sequence(seed=seed)
        )

        # Save channel waveform
        chan_file = output_dir / "standard_waveform_chan.txt"
        save_pwl_file(chan_file, t_chan, i_chan)
        waveform_files["chan"] = chan_file

        # Save enable waveform
        enab_file = output_dir / "standard_waveform_enab.txt"
        save_pwl_file(enab_file, t_enab, i_enab)
        waveform_files["enab"] = enab_file

        logger.info(
            f"   ✅ Generated standard waveforms: {chan_file.name}, {enab_file.name}"
        )

    # TODO: Add sweep and comparison waveform generation
    if "sweep" in waveform_config:
        logger.warning("   ⚠️  Sweep waveform generation not yet implemented")

    if "comparison" in waveform_config:
        logger.warning("   ⚠️  Comparison waveform generation not yet implemented")

    logger.info(
        f"✅ Waveform generation complete. Generated {len(waveform_files)} files."
    )
    return waveform_files


def run_simulation(
    config_path: Union[str, Path],
    waveform_files: Optional[Dict[str, Path]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Execute SPICE simulation with generated waveforms.

    Args:
        config_path: Path to configuration file
        waveform_files: Dictionary mapping waveform names to file paths (auto-generated if None)
        output_dir: Directory to save simulation results (defaults to standard results dir)

    Returns:
        Dictionary mapping simulation names to result file paths
    """
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)

    # Generate waveforms if not provided
    if waveform_files is None:
        waveform_files = generate_waveforms(config_path)

    if output_dir is None:
        output_dir = get_results_dir()
    ensure_directory_exists(output_dir)

    logger.info("🔬 Running SPICE simulations...")

    # Extract simulation configuration
    sim_config = config.get("simulation", {})

    # Setup LTspice runner
    ltspice_path = sim_config.get(
        "ltspice_path", "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe"
    )
    runner = LTspiceRunner(ltspice_path=ltspice_path)

    # Setup automation
    automator = SimulationAutomator(runner)

    # Get circuit file
    circuit_dir = Path(__file__).parent / "circuit_files"
    template_netlist = circuit_dir / sim_config.get(
        "template_netlist", "default_circuit_template.cir"
    )

    if not template_netlist.exists():
        raise FileNotFoundError(f"Circuit template not found: {template_netlist}")

    simulation_results = {}

    # Run single simulation if standard waveforms exist
    if "chan" in waveform_files and "enab" in waveform_files:
        # Save generated netlist in circuit_files (will be overwritten each time)
        output_netlist = circuit_dir / "generated_simulation.cir"

        result = automator.run_waveform_simulation(
            template_netlist=template_netlist,
            chan_pwl_path=waveform_files["chan"],
            enab_pwl_path=waveform_files["enab"],
            output_netlist=output_netlist,
            extract_results=False,  # We'll analyze the raw file ourselves
        )

        # The raw file will be created alongside the netlist in circuit_files
        raw_file_in_circuit_dir = output_netlist.with_suffix(".raw")
        
        if raw_file_in_circuit_dir.exists():
            # Copy the result to the results directory  
            result_file = output_dir / "simulation.raw"
            import shutil
            shutil.copy2(raw_file_in_circuit_dir, result_file)
            
            simulation_results["standard"] = result_file
            logger.info(f"   ✅ Standard simulation complete: {result_file.name}")
            
            # Optionally remove the raw file from circuit_files to keep it clean
            # raw_file_in_circuit_dir.unlink()  # Uncomment to auto-delete
        else:
            logger.error(f"   ❌ Simulation failed - no output file: {raw_file_in_circuit_dir}")

    # Run parameter sweeps if configured
    if "sweep" in config.get("simulation", {}):
        logger.warning("   ⚠️  Parameter sweep simulation not yet implemented")

    logger.info(
        f"✅ Simulation complete. Generated {len(simulation_results)} result files."
    )
    return simulation_results


def plot_results(
    config_path: Union[str, Path],
    simulation_results: Optional[Dict[str, Path]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate analysis plots from simulation results.

    Args:
        config_path: Path to configuration file
        simulation_results: Dictionary mapping simulation names to result file paths (auto-generated if None)
        output_dir: Directory to save plots (defaults to standard plots dir)

    Returns:
        Dictionary mapping plot names to file paths
    """
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)

    # Run simulation if results not provided
    if simulation_results is None:
        simulation_results = run_simulation(config_path)

    if output_dir is None:
        output_dir = get_results_dir() / "plots"
    ensure_directory_exists(output_dir)

    logger.info("📊 Generating analysis plots...")

    # Extract plotting configuration
    plot_config = config.get("plotting", {})

    # Initialize plotter
    plotting_config_file = get_config_dir() / "plotting_config.yaml"
    if plotting_config_file.exists():
        plotter = UnifiedPlotter(config_path=plotting_config_file)
    else:
        plotter = UnifiedPlotter()

    plot_files = {}

    # Generate plots for each simulation result
    for sim_name, result_file in simulation_results.items():
        logger.info(f"   📈 Processing {sim_name}: {result_file.name}")

        # Load simulation data
        try:
            import ltspice

            ltsp = ltspice.Ltspice(str(result_file))
            ltsp.parse()

            # Extract data for plotting
            data = plotter.extract_simulation_data(ltsp)

            # Generate requested plot types
            plot_types = plot_config.get("types", ["transient", "multi_panel"])

            for plot_type_name in plot_types:
                try:
                    plot_type = PlotType[plot_type_name.upper()]
                    output_file = output_dir / f"{sim_name}_{plot_type_name}.png"

                    fig = plotter.create_plot(plot_type, data, str(output_file))
                    if fig is not None:
                        plot_files[f"{sim_name}_{plot_type_name}"] = output_file
                        logger.info(
                            f"      ✅ {plot_type_name} plot: {output_file.name}"
                        )

                except (KeyError, Exception) as e:
                    logger.warning(
                        f"      ⚠️  Failed to create {plot_type_name} plot: {e}"
                    )

        except Exception as e:
            logger.error(f"   ❌ Failed to process {sim_name}: {e}")

    # Generate comprehensive report if requested
    if plot_config.get("generate_report", False):
        try:
            report_dir = plotter.generate_report(
                simulation_results, output_dir / "comprehensive_report"
            )
            plot_files["report"] = report_dir
            logger.info(f"   ✅ Comprehensive report: {report_dir}")
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to generate comprehensive report: {e}")

    logger.info(f"✅ Plotting complete. Generated {len(plot_files)} plots.")
    return plot_files


def run_complete_workflow(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Run the complete simulation workflow.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing all generated files and results
    """
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting complete SPICE simulation workflow...")

    # Step 1: Generate waveforms
    waveform_files = generate_waveforms(config_path)

    # Step 2: Run simulations (pass the waveforms to avoid regenerating)
    simulation_results = run_simulation(config_path, waveform_files)

    # Step 3: Generate plots
    plot_files = plot_results(config_path)

    # Load configuration for results
    config = load_config(config_path)

    # Compile results
    results = {
        "waveforms": waveform_files,
        "simulations": simulation_results,
        "plots": plot_files,
        "config": config,
    }

    logger.info("🎉 Complete workflow finished successfully!")
    logger.info(f"   📊 Generated {len(waveform_files)} waveforms")
    logger.info(f"   🔬 Ran {len(simulation_results)} simulations")
    logger.info(f"   📈 Created {len(plot_files)} plots")

    return results


def create_default_config(output_path: Union[str, Path]) -> Path:
    """Create a default configuration file.

    Args:
        output_path: Path where to save the configuration file

    Returns:
        Path to created configuration file
    """
    output_path = Path(output_path)

    default_config = {
        "waveforms": {
            "standard": {
                "cycle_time": 1e-6,
                "pulse_sigma": 35e-9,
                "hold_width_write": 120e-9,
                "hold_width_read": 300e-9,
                "hold_width_clear": 5e-9,
                "write_amplitude": 80e-6,
                "read_amplitude": 725e-6,
                "enab_write_amplitude": 465e-6,
                "enab_read_amplitude": 300e-6,
                "clear_amplitude": 700e-6,
                "dt": 0.1e-9,
                "seed": 42,
            }
        },
        "simulation": {
            "template": "nmem_cell_read_template.cir",
            "ltspice_path": "/mnt/c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
            "parameters": {"temp": 4.2, "timeout": 300},
        },
        "plotting": {
            "types": ["transient", "multi_panel"],
            "generate_report": True,
            "style": "publication",
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    return output_path


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="SPICE Simulation Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete workflow with config file
    python spice_simulation.py --config simulation_config.yaml
    
    # Create default config
    python spice_simulation.py --create-config default_config.yaml
    
    # Run specific steps
    python spice_simulation.py --config config.yaml --step waveforms
    python spice_simulation.py --config config.yaml --step simulation  
    python spice_simulation.py --config config.yaml --step plotting
        """,
    )

    parser.add_argument("--config", type=str, help="Configuration file path")

    parser.add_argument(
        "--create-config",
        type=str,
        metavar="OUTPUT_PATH",
        help="Create default configuration file",
    )

    parser.add_argument(
        "--step",
        choices=["waveforms", "simulation", "plotting", "all"],
        default="all",
        help="Run specific workflow step (default: all)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Create default config if requested
        if args.create_config:
            config_path = create_default_config(args.create_config)
            logger.info(f"✅ Created default configuration: {config_path}")
            return

        # Require config for other operations
        if not args.config:
            logger.error(
                "❌ Configuration file required. Use --config or --create-config"
            )
            sys.exit(1)

        # Load configuration
        config = load_config(args.config)

        # Run requested workflow steps
        if args.step == "all":
            run_complete_workflow(args.config)
        elif args.step == "waveforms":
            generate_waveforms(args.config)
        elif args.step == "simulation":
            run_simulation(args.config)
        elif args.step == "plotting":
            plot_results(args.config)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
