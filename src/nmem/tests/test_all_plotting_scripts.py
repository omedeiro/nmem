import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inspect
import importlib

# Replace with your actual module path
script_module = importlib.import_module('nmem.scripts')

def test_all_plot_functions_run():
    plot_functions = [
        func for name, func in inspect.getmembers(script_module, inspect.isfunction)
        if name.startswith('plot_')
    ]
    for plot_func in plot_functions:
        # Try to call with a dummy Axes if the first argument is 'ax'
        args = inspect.getfullargspec(plot_func).args
        if args and args[0] == 'ax':
            fig, ax = plt.subplots()
            try:
                plot_func(ax, {})  # Pass an empty dict or minimal dummy data
            except Exception as e:
                raise AssertionError(f"{plot_func.__name__} failed: {e}")
        else:
            try:
                plot_func({})  # Or adjust as needed for your function signatures
            except Exception as e:
                raise AssertionError(f"{plot_func.__name__} failed: {e}")