# Global Plot Style Configuration

This system allows you to globally switch between different plotting styles for presentations, papers, and thesis writing across all plotting scripts in the nmem project.

## Available Styles

1. **Presentation Style** (`"presentation"` or `"pres"`)
   - Large fonts for visibility
   - Inter font family
   - Higher DPI (600)
   - Larger figure sizes (6x4 inches)
   - Suitable for talks and presentations

2. **Paper Style** (`"paper"` or `"publication"`) - **Default**
   - Compact layout with serif fonts
   - Smaller figure sizes (3.5 inches wide, golden ratio)
   - Publication-ready formatting
   - Suitable for academic papers

3. **Thesis Style** (`"thesis"`)
   - Balanced between presentation and paper styles
   - Medium figure sizes (5 inches wide)
   - Inter font family
   - Moderate DPI (300)
   - Suitable for thesis writing

## Usage

### Method 1: Using run_all_plotting_scripts.py (Recommended)

Generate all plots with a specific style:

```bash
# Generate all plots with thesis style (default)
python run_all_plotting_scripts.py

# Generate all plots with presentation style
python run_all_plotting_scripts.py --style presentation

# Generate all plots with paper style  
python run_all_plotting_scripts.py --style paper

# Specify custom output directory and style
python run_all_plotting_scripts.py ./my_plots --style thesis
```

### Method 2: Programmatic Style Setting

In your Python scripts:

```python
from nmem.analysis.styles import set_style_mode, apply_global_style

# Set global style mode
set_style_mode("presentation")  # or "paper", "thesis"

# Apply the style (call this before creating plots)
apply_global_style()

# Your plotting code here...
```

### Method 3: Using Style Configuration Helper

```python
from nmem.analysis.style_config import set_presentation_style, set_thesis_style, set_paper_style

# Quick style setters
set_presentation_style()
# or
set_thesis_style()  
# or
set_paper_style()
```

### Method 4: Command Line Style Configuration

```bash
# Set global style from command line
python -m nmem.analysis.style_config presentation
python -m nmem.analysis.style_config thesis
python -m nmem.analysis.style_config paper
```

## Updating Existing Scripts

To make existing plotting scripts use the global style system, replace style-specific calls:

**Before:**
```python
from nmem.analysis.styles import set_plot_style, set_pres_style

set_plot_style()  # or set_pres_style()
```

**After:**
```python
from nmem.analysis.styles import apply_global_style

apply_global_style()
```

## Style Customization

You can override specific style parameters:

```python
from nmem.analysis.styles import apply_global_style

# Apply global style with custom parameters
apply_global_style(dpi=450, font_size=12)
```

## Examples

### Generate all plots for a presentation:
```bash
python run_all_plotting_scripts.py --style presentation
```

### Generate all plots for your thesis:
```bash
python run_all_plotting_scripts.py --style thesis
```

### Generate all plots for a paper submission:
```bash
python run_all_plotting_scripts.py --style paper
```

### Check current style:
```python
from nmem.analysis.styles import get_style_mode
print(f"Current style: {get_style_mode()}")
```

## Style Specifications

### Presentation Style
- Figure size: 6×4 inches
- Font: Inter, 14pt base
- DPI: 600
- Line width: 2.0
- Grid: Enabled with 40% alpha

### Paper Style  
- Figure size: 3.5×2.16 inches (golden ratio)
- Font: Serif, 9pt base
- DPI: Default
- Line width: 1.2
- Compact layout

### Thesis Style
- Figure size: 5×3.09 inches (golden ratio)
- Font: Inter, 11pt base  
- DPI: 300
- Line width: 1.5
- Balanced layout

The global style system ensures consistency across all your plots while making it easy to switch between different formatting requirements.
