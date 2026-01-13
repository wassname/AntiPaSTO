"""AntiPaSTO: Self-Supervised Steering of Moral Reasoning
https://github.com/wassname/AntiPaSTO
(c) 2026 Michael J Clark, MIT License
"""
from . import control, extract
from .extract import ControlVector
from .dataset import make_dataset

__author__ = "Michael J Clark"
__url__ = "https://github.com/wassname/AntiPaSTO"
__version__ = "0.5.0"
__all__ = ["control", "extract", "ControlVector"]
