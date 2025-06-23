#!/usr/bin/env python3
"""
Script to run all plotting scripts and save outputs to the plots directory.
This provides a centralized way to generate all plots for the nmem project.
"""
import os
import sys
import importlib
import logging
import traceback
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_plotting_scripts():
    """Get list of all plotting script modules in the scripts directory."""
    scripts_dir = Path(__file__).parent
    plotting_scripts = []

    current_file = Path(__file__).name

    for file_path in scripts_dir.glob("plot_*.py"):
        if file_path.name != current_file:  # Avoid self-import
            module_name = f"nmem.scripts.{file_path.stem}"
            plotting_scripts.append((file_path.stem, module_name))

    return plotting_scripts


def run_script_with_save_dir(script_name, module_name, save_dir):
    """
    Try to run a plotting script, attempting different ways to pass save_dir.
    """
    try:
        module = importlib.import_module(module_name)

        # Method 1: Try to call main() with save_dir parameter
        if hasattr(module, "main"):
            main_func = getattr(module, "main")

            # Check if main function accepts save_dir parameter
            import inspect

            sig = inspect.signature(main_func)
            if "save_dir" in sig.parameters:
                logger.info(f"Running {script_name} with save_dir parameter")
                main_func(save_dir=save_dir)
                return True
            else:
                logger.info(f"Running {script_name} (main function without save_dir)")
                main_func()
                return True

        # Method 2: Try to call generate_plots with save_dir
        elif hasattr(module, "generate_plots"):
            logger.info(f"Running {script_name} via generate_plots")
            generate_plots = getattr(module, "generate_plots")
            # This would need data - might need to be customized per script
            return False

        # Method 3: Script might just execute on import (like plot_compare_energy_bar.py)
        else:
            logger.info(f"Running {script_name} (executes on import)")
            return True

    except Exception as e:
        logger.error(f"Error running {script_name}: {str(e)}")
        logger.debug(f"Traceback for {script_name}:\n{traceback.format_exc()}")
        return False


def modify_script_for_saving(script_name, save_dir):
    """
    Modify scripts that use plt.show() to save instead.
    This is a fallback for scripts that don't accept save_dir.
    """
    # Override plt.show() to save instead
    original_show = plt.show
    figure_counter = [0]  # Use list to make it mutable in nested function

    def save_instead_of_show():
        figure_counter[0] += 1
        if figure_counter[0] == 1:
            save_path = os.path.join(save_dir, f"{script_name}.png")
        else:
            save_path = os.path.join(
                save_dir, f"{script_name}_fig{figure_counter[0]}.png"
            )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close("all")  # Close all figures to free memory
        logger.info(f"Saved plot to {save_path}")

    plt.show = save_instead_of_show
    return original_show


def main(output_dir=None):
    """
    Main function to run all plotting scripts.

    Args:
        output_dir (str): Directory to save plots. Defaults to '../plots'
    """
    if output_dir is None:
        # Default to plots directory relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "src" / "nmem" / "plots"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving plots to: {output_dir}")

    # Get all plotting scripts
    plotting_scripts = get_plotting_scripts()
    logger.info(f"Found {len(plotting_scripts)} plotting scripts")

    successful_runs = 0
    failed_runs = 0

    for script_name, module_name in plotting_scripts:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {script_name}")
        logger.info(f"{'='*50}")

        # Store original plt.show
        original_show = modify_script_for_saving(script_name, str(output_dir))

        try:
            success = run_script_with_save_dir(
                script_name, module_name, str(output_dir)
            )
            if success:
                successful_runs += 1
                logger.info(f"✓ Successfully processed {script_name}")
            else:
                failed_runs += 1
                logger.warning(f"✗ Failed to process {script_name}")

        except Exception as e:
            failed_runs += 1
            logger.error(f"✗ Exception in {script_name}: {str(e)}")

        finally:
            # Restore original plt.show
            plt.show = original_show
            # Close any remaining figures
            plt.close("all")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total scripts: {len(plotting_scripts)}")
    logger.info(f"Successful: {successful_runs}")
    logger.info(f"Failed: {failed_runs}")
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    # Allow command line argument for output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = None

    main(output_dir)
