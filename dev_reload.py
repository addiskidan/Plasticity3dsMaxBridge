import sys
import os
import importlib
try:
    from PySide6 import QtWidgets
except ImportError:
    from PySide2 import QtWidgets
    
PLUGIN_DIR = r"C:\3dsmax_dev\PlasticityLiveLink_Max"

if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

def close_existing():
    """Close existing Plasticity UI instances"""
    app = QtWidgets.QApplication.instance()
    for widget in app.topLevelWidgets():
        if widget.objectName() in ["PlasticityUI", "PlasticityDock"]:
            widget.close()
            widget.deleteLater()
    # Also check for dock widgets
    for child in app.allWidgets():
        if isinstance(child, QtWidgets.QDockWidget) and child.objectName() == "PlasticityDock":
            child.close()
            child.deleteLater()

MODULES = ["client", "handler", "plasticity_ui"]

for name in MODULES:
    if name in sys.modules:
        importlib.reload(sys.modules[name])
    else:
        importlib.import_module(name)

# Launch UI
try:
    from plasticity_ui import PlasticityUI
    
    close_existing()
    
    # Create UI instance - it will handle docking itself
    win = PlasticityUI()
    
except Exception as e:
    print("Failed to launch Plasticity UI:", e)
    raise  # Re-raise the exception for debugging