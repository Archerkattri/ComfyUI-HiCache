"""ComfyUI-HiCache — training-free Hunyuan3D acceleration for ComfyUI.

Skips DiT forwards during flow-matching sampling and forecasts the velocity
instead, using the HiCache (Hermite polynomial, arXiv:2508.16984) and
HiCache++ (DMD / Prony exponential) forecasters from the ``hicache-pp``
library.

Repository: https://github.com/Archerkattri/ComfyUI-HiCache
License: MIT
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:  # imported outside ComfyUI (tests / tooling), not as a package
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
