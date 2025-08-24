"""
PyColour Extract - Advanced Color Extraction and Analysis Tool

A comprehensive tool for extracting, analyzing, and manipulating colors from images
with support for multiple algorithms, export formats, and advanced color theory.
"""

__version__ = "2.0.0"
__author__ = "PyColour Extract Team"
__license__ = "MIT"

from .core.color_extractor import ColorExtractor
from .core.color_analyzer import ColorAnalyzer
from .core.palette_generator import PaletteGenerator
from .models.color_data import ColorData, PaletteData, AnalysisResult

__all__ = [
    "ColorExtractor",
    "ColorAnalyzer", 
    "PaletteGenerator",
    "ColorData",
    "PaletteData",
    "AnalysisResult",
]
