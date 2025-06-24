#!/usr/bin/env python3
"""
Script to run all plotting scripts and save outputs to the plots directory.
This provides a centralized way to generate all plots for the nmem project.

Usage:
    python run_all_plotting_scripts.py [output_dir] [--style {presentation,paper,thesis}]

Examples:
    python run_all_plotting_scripts.py --style presentation
    python run_all_plotting_scripts.py ./plots --style thesis
    python run_all_plotting_scripts.py ./plots --style paper
"""
import argparse
import importlib
import logging
import os
import sys
import traceback
from pathlib import Path
import ast
import inspect
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import style management functions
from nmem.analysis.styles import set_style_mode, apply_global_style, get_style_mode

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
        # Apply global style before importing the module
        apply_global_style()

        module = importlib.import_module(module_name)

        # Method 1: Try to call main() with save_dir parameter
        if hasattr(module, "main"):
            main_func = getattr(module, "main")

            # Check if main function accepts save_dir parameter
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


def extract_plot_description(module_path):
    """
    Extract plot description from the module's docstring or main function docstring.

    Args:
        module_path (Path): Path to the Python module file

    Returns:
        str: Description of the plot or "No description available"
    """
    try:
        # First try to get the module docstring
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the AST to get docstrings
        tree = ast.parse(content)

        # Check module-level docstring
        if ast.get_docstring(tree):
            return ast.get_docstring(tree).strip()

        # Check main function docstring
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "main"
                and ast.get_docstring(node)
            ):
                return ast.get_docstring(node).strip()

        # If no docstring found, try to import and get runtime docstring
        module_name = f"nmem.scripts.{module_path.stem}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "__doc__") and module.__doc__:
                return module.__doc__.strip()
            elif (
                hasattr(module, "main")
                and hasattr(module.main, "__doc__")
                and module.main.__doc__
            ):
                return module.main.__doc__.strip()
        except:
            pass

        return "No description available"

    except Exception as e:
        logger.warning(f"Could not extract description from {module_path}: {e}")
        return "No description available"


def generate_plots_readme(plotting_scripts, output_dir, style_mode):
    """
    Generate a README.md file documenting all plots.

    Args:
        plotting_scripts (list): List of (script_name, module_name) tuples
        output_dir (Path): Directory where plots are saved
        style_mode (str): Current plot style mode
    """
    scripts_dir = Path(__file__).parent
    readme_path = output_dir / "README.md"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# nmem Plots Documentation\n\n")
        f.write(
            "This directory contains plots generated by the nmem project plotting scripts.\n\n"
        )
        f.write(f"**Plot Style:** {style_mode}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Plots Overview\n\n")

        # Sort scripts alphabetically
        sorted_scripts = sorted(plotting_scripts, key=lambda x: x[0])

        for script_name, module_name in sorted_scripts:
            script_path = scripts_dir / f"{script_name}.py"
            description = extract_plot_description(script_path)

            # Generate expected image filename(s)
            # Most scripts will generate script_name.png, but some may generate multiple
            image_files = []
            for ext in [".png", ".pdf", ".svg"]:
                base_image = output_dir / f"{script_name}{ext}"
                if base_image.exists():
                    image_files.append(f"{script_name}{ext}")

            # Also check for numbered figures
            fig_num = 2
            while True:
                fig_image = output_dir / f"{script_name}_fig{fig_num}.png"
                if fig_image.exists():
                    image_files.append(f"{script_name}_fig{fig_num}.png")
                    fig_num += 1
                else:
                    break

            if not image_files:
                # Default expected filename
                image_files = [f"{script_name}.png"]

            f.write(f"### {script_name}\n\n")
            f.write(f"**Script:** `{script_name}.py`\n\n")
            f.write(f"**Description:** {description}\n\n")

            # Write image references
            if len(image_files) == 1:
                f.write(f"**Image:** ![{script_name}]({image_files[0]})\n\n")
            else:
                f.write("**Images:**\n")
                for i, img_file in enumerate(image_files, 1):
                    f.write(f"- Figure {i}: ![{script_name}_fig{i}]({img_file})\n")
                f.write("\n")

            f.write("---\n\n")

        f.write("## Script Execution\n\n")
        f.write("All plots were generated using:\n\n")
        f.write("```bash\n")
        f.write(f"python run_all_plotting_scripts.py --style {style_mode}\n")
        f.write("```\n\n")
        f.write(
            "Individual plots can be regenerated by running their respective scripts in the `src/nmem/scripts/` directory.\n"
        )

    logger.info(f"Generated plots documentation: {readme_path}")


def main(output_dir=None, style_mode="thesis", generate_readme=True):
    """
    Main function to run all plotting scripts.

    Args:
        output_dir (str): Directory to save plots. Defaults to '../plots'
        style_mode (str): Global plotting style mode ('presentation', 'paper', or 'thesis')
        generate_readme (bool): Whether to generate a README.md file documenting the plots
    """
    # Set global style mode
    set_style_mode(style_mode)
    logger.info(f"Using global plot style: {get_style_mode()}")

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

    # Generate README.md if requested
    if generate_readme:
        generate_plots_readme(plotting_scripts, output_dir, style_mode)

    # Generate README documentation
    if generate_readme:
        try:
            generate_plots_readme(plotting_scripts, output_dir, style_mode)
        except Exception as e:
            logger.error(f"Failed to generate plots README: {e}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total scripts: {len(plotting_scripts)}")
    logger.info(f"Successful: {successful_runs}")
    logger.info(f"Failed: {failed_runs}")
    logger.info(f"Plot style: {get_style_mode()}")
    logger.info(f"Plots saved to: {output_dir}")
    if generate_readme:
        logger.info(f"Documentation: {output_dir}/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all plotting scripts with configurable output directory and style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_plotting_scripts.py --style presentation
  python run_all_plotting_scripts.py ./plots --style thesis  
  python run_all_plotting_scripts.py ./plots --style paper
        """,
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for plots (default: auto-detected plots directory)",
    )

    parser.add_argument(
        "--style",
        choices=["presentation", "pres", "paper", "publication", "thesis"],
        default="paper",
        help="Global plotting style mode (default: paper)",
    )

    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip generating the plots README documentation",
    )

    args = parser.parse_args()
    generate_readme = not args.no_readme
    main(args.output_dir, args.style, generate_readme)
