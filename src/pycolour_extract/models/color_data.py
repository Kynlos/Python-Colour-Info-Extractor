"""Data models for color extraction and analysis."""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class ColorSpace(Enum):
    """Supported color spaces."""
    RGB = "rgb"
    HSV = "hsv"
    HSL = "hsl"
    LAB = "lab"
    LCH = "lch"
    XYZ = "xyz"
    CMYK = "cmyk"


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    XML = "xml"
    YAML = "yaml"
    ASE = "ase"  # Adobe Swatch Exchange
    ACO = "aco"  # Adobe Color
    GPL = "gpl"  # GIMP Palette
    SCSS = "scss"
    CSS = "css"
    LESS = "less"
    STYLUS = "stylus"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    JAVA = "java"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    SVG = "svg"
    PNG = "png"
    PDF = "pdf"


@dataclass
class ColorInfo:
    """Individual color information."""
    rgb: Tuple[int, int, int]
    hex: str
    name: Optional[str] = None
    frequency: int = 0
    percentage: float = 0.0
    hsv: Optional[Tuple[float, float, float]] = None
    hsl: Optional[Tuple[float, float, float]] = None
    lab: Optional[Tuple[float, float, float]] = None
    cmyk: Optional[Tuple[float, float, float, float]] = None
    luminance: Optional[float] = None
    contrast_ratio_white: Optional[float] = None
    contrast_ratio_black: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rgb": self.rgb,
            "hex": self.hex,
            "name": self.name,
            "frequency": self.frequency,
            "percentage": self.percentage,
            "hsv": self.hsv,
            "hsl": self.hsl,
            "lab": self.lab,
            "cmyk": self.cmyk,
            "luminance": self.luminance,
            "contrast_ratio_white": self.contrast_ratio_white,
            "contrast_ratio_black": self.contrast_ratio_black,
        }


@dataclass
class ColorHarmony:
    """Color harmony information."""
    harmony_type: str
    base_color: ColorInfo
    harmony_colors: List[ColorInfo]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "harmony_type": self.harmony_type,
            "base_color": self.base_color.to_dict(),
            "harmony_colors": [c.to_dict() for c in self.harmony_colors],
            "confidence": self.confidence,
        }


@dataclass
class ColorCluster:
    """Color clustering information."""
    cluster_id: int
    centroid: ColorInfo
    colors: List[ColorInfo]
    size: int
    variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid.to_dict(),
            "colors": [c.to_dict() for c in self.colors],
            "size": self.size,
            "variance": self.variance,
        }


@dataclass
class ImageMetadata:
    """Image metadata information."""
    path: str
    filename: str
    size: Tuple[int, int]
    format: str
    mode: str
    total_pixels: int
    file_size: int
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "filename": self.filename,
            "size": self.size,
            "format": self.format,
            "mode": self.mode,
            "total_pixels": self.total_pixels,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }


@dataclass
class ColorData:
    """Complete color extraction data for an image."""
    image_metadata: ImageMetadata
    colors: List[ColorInfo]
    unique_color_count: int
    dominant_color: ColorInfo
    average_color: ColorInfo
    clusters: Optional[List[ColorCluster]] = None
    harmonies: Optional[List[ColorHarmony]] = None
    accessibility_score: Optional[float] = None
    color_temperature: Optional[float] = None
    vibrance: Optional[float] = None
    saturation_distribution: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "image_metadata": self.image_metadata.to_dict(),
            "colors": [c.to_dict() for c in self.colors],
            "unique_color_count": self.unique_color_count,
            "dominant_color": self.dominant_color.to_dict(),
            "average_color": self.average_color.to_dict(),
            "accessibility_score": self.accessibility_score,
            "color_temperature": self.color_temperature,
            "vibrance": self.vibrance,
            "saturation_distribution": self.saturation_distribution,
        }
        
        if self.clusters:
            data["clusters"] = [c.to_dict() for c in self.clusters]
        if self.harmonies:
            data["harmonies"] = [h.to_dict() for h in self.harmonies]
            
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


@dataclass
class PaletteData:
    """Color palette data."""
    name: str
    colors: List[ColorInfo]
    palette_type: str
    source_image: Optional[str] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "colors": [c.to_dict() for c in self.colors],
            "palette_type": self.palette_type,
            "source_image": self.source_image,
            "created_at": self.created_at,
        }


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    color_data: ColorData
    palettes: List[PaletteData]
    processing_time: float
    algorithm_used: str
    settings: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "color_data": self.color_data.to_dict(),
            "palettes": [p.to_dict() for p in self.palettes],
            "processing_time": self.processing_time,
            "algorithm_used": self.algorithm_used,
            "settings": self.settings,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)
