#!/usr/bin/env python3
"""
Update plots README based on existing files and script descriptions.
This script creates documentation for all plots in the plots directory,
extracting descriptions from the plotting scripts where available.

Usage:
    python update_plots_readme.py [plots_directory] [--style paper]

Examples:
    python update_plots_readme.py
    python update_plots_readme.py /path/to/plots --style thesis
"""
import argparse
import ast
import logging
import sys
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_plot_description(script_path):
    """
    Extract plot description from a script's docstring.

    Args:
        script_path (Path): Path to the Python script file

    Returns:
        str: Description of the plot or "No description available"
    """
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the AST to get docstrings
        tree = ast.parse(content)

        # Check module-level docstring first
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

        return "No description available"

    except Exception as e:
        logger.warning(f"Could not extract description from {script_path.name}: {e}")
        return "No description available"


def find_script_for_image(image_name, scripts_dir):
    """
    Try to find the most likely script that generated an image.

    Args:
        image_name (str): Name of the image file (without extension)
        scripts_dir (Path): Directory containing the plotting scripts

    Returns:
        Path or None: Path to the most likely script file
    """
    # Direct match: plot_something.png -> plot_something.py
    direct_script = scripts_dir / f"{image_name}.py"
    if direct_script.exists():
        return direct_script

    # Try adding plot_ prefix: something.png -> plot_something.py
    if not image_name.startswith("plot_"):
        prefixed_script = scripts_dir / f"plot_{image_name}.py"
        if prefixed_script.exists():
            return prefixed_script

    # For complex cases, look for partial matches
    image_base = image_name.replace("plot_", "")
    for script_file in scripts_dir.glob("plot_*.py"):
        script_base = script_file.stem.replace("plot_", "")
        if script_base in image_base or image_base in script_base:
            return script_file

    return None


def create_plots_readme(plots_dir, style_mode="paper"):
    """
    Create a comprehensive README.md file for all plots in the directory.

    Args:
        plots_dir (Path): Directory containing the plots
        style_mode (str): Style mode used for plot generation
    """
    plots_dir = Path(plots_dir)
    scripts_dir = (
        plots_dir.parent / "scripts"
    )  # plots is in src/nmem/plots, scripts is in src/nmem/scripts
    readme_path = plots_dir / "README.md"

    # Get all PNG files in the directory
    png_files = sorted(plots_dir.glob("*.png"))

    if not png_files:
        logger.warning(f"No PNG files found in {plots_dir}")
        return

    # Group files by their likely script origin
    script_groups = {}

    for png_file in png_files:
        image_name = png_file.stem

        # Special case: Group all array parameter matrix plots together
        if image_name.startswith("array_") and image_name.endswith("_matrix"):
            script_name = "plot_array_parameter_matrix"
            script_file = scripts_dir / f"{script_name}.py"
        else:
            # Try to find the corresponding script
            script_file = find_script_for_image(image_name, scripts_dir)

            if script_file:
                script_name = script_file.stem
            else:
                # Fallback: guess the script name
                if image_name.startswith("plot_"):
                    script_name = image_name
                elif image_name.startswith("ber_"):
                    script_name = f"plot_{image_name}"
                else:
                    script_name = f"plot_{image_name}"

        # Use script_name as key, not image_name
        if script_name not in script_groups:
            script_groups[script_name] = {
                "script_file": (
                    script_file if script_file and script_file.exists() else None
                ),
                "images": [],
            }
        script_groups[script_name]["images"].append(png_file.name)

    # Write the README
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# nmem Plots Documentation\n\n")
        f.write(
            "This directory contains plots generated by the nmem project plotting scripts.\n\n"
        )
        f.write(f"**Plot Style:** {style_mode}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Plots:** {len(png_files)}\n\n")
        f.write("## Plots Overview\n\n")

        # Sort by script name
        for script_name in sorted(script_groups.keys()):
            group = script_groups[script_name]
            image_files = sorted(group["images"])

            # Extract description if script file exists
            if group["script_file"] and group["script_file"].exists():
                description = extract_plot_description(group["script_file"])
            else:
                description = "No description available"

            f.write(f"### {script_name}\n\n")
            f.write(f"**Script:** `{script_name}.py`\n\n")
            f.write(f"**Description:** {description}\n\n")

            # Special handling for array parameter matrix plots
            if script_name == "plot_array_parameter_matrix" and len(image_files) > 1:
                f.write("**Array Parameter Matrix Plots:**\n\n")

                # Create table with 3 columns
                f.write("| Parameter | Plot | Parameter | Plot |\n")
                f.write("|-----------|------|-----------|------|\n")

                # Group images in pairs for table formatting
                for i in range(0, len(image_files), 2):
                    left_img = image_files[i]
                    right_img = image_files[i + 1] if i + 1 < len(image_files) else None

                    # Extract parameter names from filenames (array_PARAMETER_matrix.png)
                    left_param = (
                        left_img.replace("array_", "")
                        .replace("_matrix.png", "")
                        .replace("_", " ")
                        .title()
                    )

                    left_cell = f"{left_param} | ![{left_param}]({left_img})"

                    if right_img:
                        right_param = (
                            right_img.replace("array_", "")
                            .replace("_matrix.png", "")
                            .replace("_", " ")
                            .title()
                        )
                        right_cell = f"{right_param} | ![{right_param}]({right_img})"
                    else:
                        right_cell = " | "

                    f.write(f"| {left_cell} | {right_cell} |\n")

                f.write("\n")
            else:
                # Standard handling for other plots
                if len(image_files) == 1:
                    f.write(f"![{script_name}]({image_files[0]})\n\n")
                else:
                    f.write("**Images:**\n")
                    for i, img_file in enumerate(image_files, 1):
                        f.write(f"- ![{script_name}_fig{i}]({img_file})\n")
                    f.write("\n")

            f.write("---\n\n")

        f.write("## Usage\n\n")
        f.write("**Regenerate all plots:**\n")
        f.write("```bash\n")
        f.write(f"python run_all_plotting_scripts.py --style {style_mode}\n")
        f.write("```\n\n")
        f.write("**Update this documentation:**\n")
        f.write("```bash\n")
        f.write("python update_plots_readme.py\n")
        f.write("```\n\n")
        f.write(
            "**Individual plots:** Run the corresponding script in `src/nmem/scripts/`\n"
        )

    logger.info(f"Created {readme_path}")
    logger.info(
        f"Documented {len(png_files)} plots organized into {len(script_groups)} script groups"
    )


def main():
    """Main function to update plots README."""
    parser = argparse.ArgumentParser(
        description="Update plots README based on existing files and script descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "plots_dir",
        nargs="?",
        help="Directory containing plots (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--style",
        default="paper",
        choices=["presentation", "paper", "thesis"],
        help="Plot style mode for documentation (default: paper)",
    )

    args = parser.parse_args()

    # Determine plots directory
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        # Auto-detect: assume this script is in src/nmem/scripts/
        script_dir = Path(__file__).parent
        plots_dir = script_dir.parent / "plots"

    if not plots_dir.exists():
        logger.error(f"Plots directory {plots_dir} does not exist")
        return 1

    logger.info(f"Updating README for plots in: {plots_dir}")

    try:
        create_plots_readme(plots_dir, args.style)
        return 0
    except Exception as e:
        logger.error(f"Error creating README: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
