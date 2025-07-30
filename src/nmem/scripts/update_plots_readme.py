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

    # Search through script contents to find which script generates this image
    for script_file in scripts_dir.glob("plot_*.py"):
        try:
            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Look for savefig calls that might generate this image
                if f'"{image_name}.png"' in content or f"'{image_name}.png'" in content:
                    return script_file
                # Also check with f-strings
                if f'{image_name}.png' in content:
                    return script_file
        except Exception:
            continue

    # For complex cases, look for partial matches
    image_base = image_name.replace("plot_", "")
    for script_file in scripts_dir.glob("plot_*.py"):
        script_base = script_file.stem.replace("plot_", "")
        if script_base in image_base or image_base in script_base:
            return script_file

    return None


def extract_figure_size_from_script(script_file):
    """
    Extract the figure size from a plotting script.

    Args:
        script_file (Path): Path to the script file

    Returns:
        tuple: (width, height) in inches, or None if not found
    """
    if not script_file or not script_file.exists():
        return None

    try:
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for common figure size patterns
        patterns = [
            # Direct figsize specification: figsize=(6, 4)
            r"figsize\s*=\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
            # Division patterns: figsize=(120/25.4, 90/25.4)
            r"figsize\s*=\s*\(\s*([0-9.]+)\s*/\s*[0-9.]+\s*,\s*([0-9.]+)\s*/\s*[0-9.]+\s*\)",
            # get_consistent_figure_size patterns
            r'get_consistent_figure_size\(["\'](\w+)["\']\)',
        ]

        import re

        for pattern in patterns[:2]:  # Direct figsize patterns
            match = re.search(pattern, content)
            if match:
                try:
                    width = float(match.group(1))
                    height = float(match.group(2))
                    # Handle division cases like 120/25.4
                    if "/" in pattern:
                        # For mm to inch conversion (25.4 mm = 1 inch)
                        if "25.4" in match.group(0):
                            width = width / 25.4
                            height = height / 25.4
                    return (width, height)
                except (ValueError, IndexError):
                    continue

        # Handle get_consistent_figure_size cases
        get_size_match = re.search(patterns[2], content)
        if get_size_match:
            plot_type = get_size_match.group(1)
            # Base size from styles.py
            base_width, base_height = 6, 4

            size_map = {
                "single": (base_width, base_height),
                "wide": (base_width * 3, base_height * 2),
                "tall": (base_width, base_height * 1.5),
                "square": (base_width, base_width),
                "comparison": (base_width * 2, base_height),
                "grid": (base_width * 2, base_height * 2),
                "multi_row": (base_width, base_height * 1.5),
                "large": (base_width * 2, base_height * 4),
            }
            return size_map.get(plot_type, (base_width, base_height))

        # Default fallback to base size
        return (6, 4)

    except Exception as e:
        logger.warning(f"Could not extract figure size from {script_file.name}: {e}")
        return None


def format_image_with_width(image_name, alt_text, figure_size=None, in_table=False):
    """
    Format an image reference with width control when possible.

    Args:
        image_name (str): Image filename
        alt_text (str): Alt text for the image
        figure_size (tuple): (width, height) in inches, or None
        in_table (bool): Whether the image will be displayed in a table

    Returns:
        str: Formatted image reference
    """
    # For images in tables, use standard markdown to avoid table formatting issues
    if in_table:
        return f"![{alt_text}]({image_name})"
    
    # For standalone images, use HTML with width when available
    # This works on GitHub and most modern markdown viewers
    if figure_size and len(figure_size) == 2:
        width_inches = figure_size[0]
        # Convert to pixels (assume 96 DPI for web display)
        width_pixels = int(width_inches * 96)
        # Cap at reasonable size for web viewing
        if width_pixels > 800:
            width_pixels = 800
        return f'<img src="{image_name}" alt="{alt_text}" width="{width_pixels}">'
    else:
        # Use standard markdown if no size info available
        return f"![{alt_text}]({image_name})"


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
        # Special case: Group voltage trace toggle write enable plots together
        elif (
            image_name.startswith("voltage_")
            and ("_off" in image_name or "_on" in image_name)
            and ("trace_stack" in image_name or "write_current_sweep" in image_name)
        ):
            script_name = "plot_voltage_trace_toggle_write_enable"
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
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Plots Overview\n\n")

        # Sort by script name
        for script_name in sorted(script_groups.keys()):
            group = script_groups[script_name]
            image_files = sorted(group["images"])

            # Extract description if script file exists
            if group["script_file"] and group["script_file"].exists():
                description = extract_plot_description(group["script_file"])
                figure_size = extract_figure_size_from_script(group["script_file"])
            else:
                description = "No description available"
                figure_size = None

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

                    left_cell = f"{left_param} | {format_image_with_width(left_img, left_param, figure_size, in_table=True)}"

                    if right_img:
                        right_param = (
                            right_img.replace("array_", "")
                            .replace("_matrix.png", "")
                            .replace("_", " ")
                            .title()
                        )
                        right_cell = f"{right_param} | {format_image_with_width(right_img, right_param, figure_size, in_table=True)}"
                    else:
                        right_cell = " | "

                    f.write(f"| {left_cell} | {right_cell} |\n")

                f.write("\n")
            # Special handling for supplemental plots with multiple parts
            elif (
                script_name == "plot_ber_enable_write_sweep_supplemental"
                and len(image_files) > 1
            ):
                f.write("**BER Enable Write Sweep Supplemental:**\n\n")

                # Create table for part1 and part2
                f.write("| Part 1 | Part 2 |\n")
                f.write("|--------|--------|\n")

                # Find part1 and part2 images
                part1_img = next((img for img in image_files if "part1" in img), None)
                part2_img = next((img for img in image_files if "part2" in img), None)

                if part1_img and part2_img:
                    f.write(
                        f"| {format_image_with_width(part1_img, 'Part 1', figure_size, in_table=True)} | {format_image_with_width(part2_img, 'Part 2', figure_size, in_table=True)} |\n"
                    )
                else:
                    # Fallback to first two images if naming pattern doesn't match
                    if len(image_files) >= 2:
                        f.write(
                            f"| {format_image_with_width(image_files[0], 'Part 1', figure_size, in_table=True)} | {format_image_with_width(image_files[1], 'Part 2', figure_size, in_table=True)} |\n"
                        )

                f.write("\n")
            # Special handling for voltage trace toggle write enable plots (2x2 table)
            elif (
                script_name == "plot_voltage_trace_toggle_write_enable"
                and len(image_files) == 4
            ):
                f.write("**Voltage Trace Toggle Write Enable (2x2 Layout):**\n\n")

                # Expected file patterns based on the script
                # Enable OFF plots (left column): voltage_write_current_sweep_off.png, voltage_trace_stack_off.png
                # Enable ON plots (right column): voltage_write_current_sweep_on.png, voltage_trace_stack_on.png

                sweep_off = next(
                    (img for img in image_files if "sweep_off" in img), None
                )
                trace_off = next(
                    (img for img in image_files if "trace_stack_off" in img), None
                )
                sweep_on = next((img for img in image_files if "sweep_on" in img), None)
                trace_on = next(
                    (img for img in image_files if "trace_stack_on" in img), None
                )

                # Create 2x2 table
                f.write("| Enable OFF | Enable ON |\n")
                f.write("|------------|----------|\n")

                # Row 1: Current sweep plots
                if sweep_off and sweep_on:
                    f.write(
                        f"| {format_image_with_width(sweep_off, 'Write Current Sweep OFF', figure_size, in_table=True)} | {format_image_with_width(sweep_on, 'Write Current Sweep ON', figure_size, in_table=True)} |\n"
                    )

                # Row 2: Trace stack plots
                if trace_off and trace_on:
                    f.write(
                        f"| {format_image_with_width(trace_off, 'Voltage Trace Stack OFF', figure_size, in_table=True)} | {format_image_with_width(trace_on, 'Voltage Trace Stack ON', figure_size, in_table=True)} |\n"
                    )

                f.write("\n")
            else:
                # Standard handling for other plots
                if len(image_files) == 1:
                    f.write(
                        f"**Image:** {format_image_with_width(image_files[0], script_name, figure_size)}\n\n"
                    )
                else:
                    f.write("**Images:**\n")
                    for i, img_file in enumerate(image_files, 1):
                        f.write(
                            f"- Figure {i}: {format_image_with_width(img_file, f'{script_name}_fig{i}', figure_size)}\n"
                        )
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
