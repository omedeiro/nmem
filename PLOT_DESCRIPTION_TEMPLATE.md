# Plot Description Template

This template shows how to add standardized descriptions to plotting scripts for automatic README generation.

## Template Format

Add a module-level docstring at the top of each plotting script:

```python
"""
Short Descriptive Title

Detailed description of what the plot shows, what data it analyzes,
and what insights it provides. Include information about:
- What is being measured or compared
- Key relationships or trends shown
- Scientific or engineering insights gained
- Any important parameters or conditions

This description will be automatically extracted and included in the
generated plots README file.
"""
import matplotlib.pyplot as plt
# ... rest of imports

def main(save_dir=None):
    """
    Optional: Add function-level docstring for additional details.
    
    Args:
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    # ... function implementation
```

## Examples

### Energy Comparison Example
```python
"""
Energy Comparison Bar Chart

This script generates a 3D extruded bar chart comparing energy consumption
across different memory technologies. The plot visualizes energy per bit
operation for various superconducting and semiconducting memory types,
providing a clear comparison of their energy efficiency characteristics.
"""
```

### BER Analysis Example
```python
"""
Bit Error Rate vs Enable/Write Current Sweep Analysis

This script analyzes and visualizes the relationship between bit error rate (BER)
and enable/write current combinations. It generates plots showing how BER varies
with different current settings, helping to identify optimal operating parameters
for memory write operations. Includes both sweep plots and state current markers
to provide comprehensive analysis of write current dependencies.
"""
```

## Best Practices

1. **First line**: Concise title that describes the plot type and main subject
2. **Body**: 2-4 sentences explaining what the plot shows and why it's important
3. **Technical details**: Include relevant parameters, data sources, or analysis methods
4. **Scientific context**: Explain what insights or conclusions can be drawn

## Automatic README Generation

Once descriptions are added, run:

```bash
# Generate all plots with documentation
python run_all_plotting_scripts.py --style paper

# Generate only documentation (skip plot generation)
python run_all_plotting_scripts.py --style paper --no-readme
```

The script will:
1. Extract descriptions from all `plot_*.py` files
2. Generate `plots/README.md` with organized documentation
3. Include image references and script information
4. Create a comprehensive plot index
