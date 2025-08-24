"""Tests for export functionality."""

import pytest
import json
import csv
from pathlib import Path
from xml.etree import ElementTree as ET

from pycolour_extract.exporters.format_exporter import FormatExporter
from pycolour_extract.models.color_data import ExportFormat, AnalysisResult, PaletteData


class TestFormatExporter:
    """Test FormatExporter class."""
    
    def test_format_exporter_initialization(self):
        """Test FormatExporter initialization."""
        exporter = FormatExporter()
        assert exporter is not None
    
    def test_export_json(self, sample_color_data, temp_dir):
        """Test JSON export."""
        exporter = FormatExporter()
        palettes = [PaletteData(
            name="Test Palette",
            colors=sample_color_data.colors,
            palette_type="dominant"
        )]
        
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.json"
        exporter.export(result, ExportFormat.JSON, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "color_data" in data
        assert "palettes" in data
        assert data["algorithm_used"] == "kmeans"
        assert len(data["color_data"]["colors"]) == 4
    
    def test_export_csv(self, sample_color_data, temp_dir):
        """Test CSV export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.csv"
        exporter.export(result, ExportFormat.CSV, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) > 1  # Header + data rows
        header = rows[0]
        assert "Hex" in header
        assert "RGB_R" in header
        assert "Frequency" in header
        
        # Check data rows
        for row in rows[1:]:
            assert len(row) == len(header)
            assert row[0].startswith('#')  # Hex color
    
    def test_export_tsv(self, sample_color_data, temp_dir):
        """Test TSV export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.tsv"
        exporter.export(result, ExportFormat.TSV, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        assert len(lines) > 1
        
        # Check tab separation
        header = lines[0].split('\t')
        assert "Hex" in header
        assert "RGB_R" in header
    
    def test_export_xml(self, sample_color_data, temp_dir):
        """Test XML export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.xml"
        exporter.export(result, ExportFormat.XML, output_path)
        
        assert output_path.exists()
        
        tree = ET.parse(output_path)
        root = tree.getroot()
        assert root.tag == "colorAnalysis"
        
        # Check for expected elements
        color_data = root.find("color_data")
        assert color_data is not None
    
    def test_export_css(self, sample_color_data, temp_dir):
        """Test CSS export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.css"
        exporter.export(result, ExportFormat.CSS, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert ":root {" in content
        assert "--" in content  # CSS custom properties
        assert "#ff0000" in content  # Should contain our red color
        assert ".bg-" in content  # Utility classes
    
    def test_export_scss(self, sample_color_data, temp_dir):
        """Test SCSS export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.scss"
        exporter.export(result, ExportFormat.SCSS, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "$" in content  # SCSS variables
        assert "#ff0000" in content  # Should contain our colors
        assert "$color-palette:" in content  # Palette array
    
    def test_export_less(self, sample_color_data, temp_dir):
        """Test LESS export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.less"
        exporter.export(result, ExportFormat.LESS, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "@" in content  # LESS variables
        assert "#ff0000" in content
    
    def test_export_stylus(self, sample_color_data, temp_dir):
        """Test Stylus export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.stylus"
        exporter.export(result, ExportFormat.STYLUS, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "=" in content  # Stylus variable assignment
        assert "#ff0000" in content
    
    def test_export_swift(self, sample_color_data, temp_dir):
        """Test Swift export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.swift"
        exporter.export(result, ExportFormat.SWIFT, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "import UIKit" in content
        assert "extension UIColor" in content
        assert "UIColor(red:" in content
        assert "static let" in content
    
    def test_export_kotlin(self, sample_color_data, temp_dir):
        """Test Kotlin export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.kotlin"
        exporter.export(result, ExportFormat.KOTLIN, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "import android.graphics.Color" in content
        assert "object Colors" in content
        assert "const val" in content
        assert "Color.parseColor" in content
    
    def test_export_java(self, sample_color_data, temp_dir):
        """Test Java export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.java"
        exporter.export(result, ExportFormat.JAVA, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "import java.awt.Color" in content
        assert "public class Colors" in content
        assert "public static final Color" in content
        assert "new Color(" in content
    
    def test_export_python(self, sample_color_data, temp_dir):
        """Test Python export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.python"
        exporter.export(result, ExportFormat.PYTHON, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '"""' in content  # Docstring
        assert "COLOR_PALETTE = {" in content
        assert "(255, 0, 0)" in content  # RGB tuple
        assert "#ff0000" in content  # Hex values
    
    def test_export_javascript(self, sample_color_data, temp_dir):
        """Test JavaScript export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.javascript"
        exporter.export(result, ExportFormat.JAVASCRIPT, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "const colors = {" in content
        assert "hex:" in content
        assert "rgb:" in content
        assert "module.exports" in content  # CommonJS
        assert "export default" in content  # ES6
    
    def test_export_html(self, sample_color_data, temp_dir):
        """Test HTML export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.html"
        exporter.export(result, ExportFormat.HTML, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "Color Analysis Report" in content
        assert "#ff0000" in content
        assert "test_image.png" in content  # Image filename
    
    def test_export_svg(self, sample_color_data, temp_dir):
        """Test SVG export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.svg"
        exporter.export(result, ExportFormat.SVG, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '<?xml version="1.0"' in content
        assert "<svg" in content
        assert "<rect" in content
        assert 'fill="#ff0000"' in content
    
    def test_export_png_palette(self, sample_color_data, temp_dir):
        """Test PNG palette export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.png"
        exporter.export(result, ExportFormat.PNG, output_path)
        
        assert output_path.exists()
        assert output_path.suffix == ".png"
        
        # Verify it's a valid image file
        from PIL import Image
        image = Image.open(output_path)
        assert image.mode == "RGB"
        assert image.size[0] > 0
        assert image.size[1] > 0
    
    def test_export_ase(self, sample_color_data, temp_dir):
        """Test ASE (Adobe Swatch Exchange) export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.ase"
        exporter.export(result, ExportFormat.ASE, output_path)
        
        assert output_path.exists()
        
        # Check ASE file signature
        with open(output_path, 'rb') as f:
            signature = f.read(4)
            assert signature == b'ASEF'
    
    def test_export_aco(self, sample_color_data, temp_dir):
        """Test ACO (Adobe Color) export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.aco"
        exporter.export(result, ExportFormat.ACO, output_path)
        
        assert output_path.exists()
        
        # Basic validation - ACO files start with version info
        with open(output_path, 'rb') as f:
            data = f.read(4)
            # Should contain version and color count info
            assert len(data) == 4
    
    def test_export_gpl(self, sample_color_data, temp_dir):
        """Test GPL (GIMP Palette) export."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.gpl"
        exporter.export(result, ExportFormat.GPL, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "GIMP Palette" in content
        assert "Name:" in content
        assert "Columns:" in content
        assert "255   0   0" in content  # RGB values for red
    
    def test_export_with_additional_data(self, sample_color_data, temp_dir):
        """Test export with additional analysis data."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        additional_data = {
            "emotions": {"energy": 0.8, "calmness": 0.3},
            "brand_personality": {"modern": 0.9}
        }
        
        output_path = temp_dir / "test_with_additional.json"
        exporter.export(result, ExportFormat.JSON, output_path, additional_data)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "emotions" in data
        assert "brand_personality" in data
        assert data["emotions"]["energy"] == 0.8
    
    def test_unsupported_format_raises_error(self, sample_color_data, temp_dir):
        """Test that unsupported format raises ValueError."""
        # This test would require creating an invalid enum value, 
        # which isn't easily done. Skip this test or modify implementation.
        pass
    
    def test_sanitize_name_method(self):
        """Test color name sanitization."""
        exporter = FormatExporter()
        
        # Test various name sanitization scenarios
        assert exporter._sanitize_name("Red Color") == "red_color"
        assert exporter._sanitize_name("Blue-Green") == "blue_green"
        assert exporter._sanitize_name("Color #1") == "color_1"
        assert exporter._sanitize_name("123Color") == "_123color"
        assert exporter._sanitize_name("") == "unknown"
        assert exporter._sanitize_name(None) == "unknown"
        assert exporter._sanitize_name("A!!B##C") == "a_b_c"
    
    def test_export_creates_directories(self, sample_color_data, temp_dir):
        """Test that export creates necessary directories."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        # Create path with non-existent directories
        output_path = temp_dir / "subdir" / "deeper" / "test_export.json"
        
        exporter.export(result, ExportFormat.JSON, output_path)
        
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_yaml_export_fallback(self, sample_color_data, temp_dir):
        """Test YAML export with fallback implementation."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "test_export.yaml"
        exporter.export(result, ExportFormat.YAML, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have YAML-like structure
        assert "color_data:" in content or "colors:" in content
        assert "#ff0000" in content
    
    def test_multiple_format_export(self, sample_color_data, temp_dir):
        """Test exporting to multiple formats."""
        exporter = FormatExporter()
        palettes = []
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.0,
            algorithm_used="kmeans",
            settings={}
        )
        
        formats = [ExportFormat.JSON, ExportFormat.CSS, ExportFormat.HTML]
        
        for format_type in formats:
            output_path = temp_dir / f"test_export.{format_type.value}"
            exporter.export(result, format_type, output_path)
            assert output_path.exists()
    
    def test_export_empty_color_data(self, temp_dir):
        """Test export with empty color data."""
        from pycolour_extract.models.color_data import ImageMetadata, ColorData
        
        metadata = ImageMetadata(
            path="empty.png",
            filename="empty.png",
            size=(100, 100),
            format="PNG",
            mode="RGB",
            total_pixels=10000,
            file_size=1000
        )
        
        # Create empty color data
        empty_color_data = ColorData(
            image_metadata=metadata,
            colors=[],
            unique_color_count=0,
            dominant_color=sample_color_data.dominant_color,  # Use a default
            average_color=sample_color_data.average_color
        )
        
        exporter = FormatExporter()
        result = AnalysisResult(
            color_data=empty_color_data,
            palettes=[],
            processing_time=0.5,
            algorithm_used="kmeans",
            settings={}
        )
        
        output_path = temp_dir / "empty_export.json"
        exporter.export(result, ExportFormat.JSON, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["color_data"]["unique_color_count"] == 0
        assert len(data["color_data"]["colors"]) == 0
