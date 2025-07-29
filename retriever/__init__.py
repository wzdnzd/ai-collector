#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
from pathlib import Path

__all__ = []

# Automatically import all .py files in the same directory
current = Path(__file__).parent
for file in current.glob("*.py"):
    if file.name != "__init__.py":
        name = file.stem
        module = importlib.import_module(f".{name}", package=__name__)

        # Add public members from the module to __all__
        if hasattr(module, "__all__"):
            __all__.extend(module.__all__)
        else:
            __all__.extend([name for name in dir(module) if not name.startswith("_")])
