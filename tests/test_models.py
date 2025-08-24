"""Tests for data models."""

import pytest
import json
from pycolour_extract.models.color_data import (
    ColorInfo, ColorData, ImageMetadata, ColorCluster,
    ColorHarmony, PaletteData, AnalysisResult, ExportFormat
)


class TestColorInfo:
    """Test ColorInfo model."""
    
    def test_color_info_creation(self):
        """Test basic ColorInfo creation."""
        color = ColorInfo(
            rgb=(255, 0, 0),
            hex="#ff0000",
            name="red",
            frequency=100,
            percentage=50.0
        )
        
        assert color.rgb == (255, 0, 0)
        assert color.hex == "#ff0000"
        assert color.name == "red"
        assert color.frequency == 100
        assert color.percentage == 50.0
    
    def test_color_info_to_dict(self):
        """Test ColorInfo dictionary conversion."""
        color = ColorInfo(
            rgb=(255, 0, 0),
            hex="#ff0000",
            name="red",
            frequency=100,
            percentage=50.0,
            hsv=(0.0, 1.0, 1.0),
            luminance=0.2126
        )
        
        color_dict = color.to_dict()
        
        assert color_dict["rgb"] == (255, 0, 0)
        assert color_dict["hex"] == "#ff0000"
        assert color_dict["name"] == "red"
        assert color_dict["hsv"] == (0.0, 1.0, 1.0)
        assert color_dict["luminance"] == 0.2126
    
    def test_color_info_optional_fields(self):
        """Test ColorInfo with optional fields."""
        color = ColorInfo(rgb=(0, 0, 0), hex="#000000")
        
        assert color.name is None
        assert color.frequency == 0
        assert color.percentage == 0.0
        assert color.hsv is None


class TestImageMetadata:
    """Test ImageMetadata model."""
    
    def test_image_metadata_creation(self):
        """Test ImageMetadata creation."""
        metadata = ImageMetadata(
            path="/path/to/image.jpg",
            filename="image.jpg",
            size=(800, 600),
            format="JPEG",
            mode="RGB",
            total_pixels=480000,
            file_size=102400
        )
        
        assert metadata.path == "/path/to/image.jpg"
        assert metadata.filename == "image.jpg"
        assert metadata.size == (800, 600)
        assert metadata.total_pixels == 480000
    
    def test_image_metadata_to_dict(self):
        """Test ImageMetadata dictionary conversion."""
        metadata = ImageMetadata(
            path="/path/to/image.jpg",
            filename="image.jpg",
            size=(800, 600),
            format="JPEG",
            mode="RGB",
            total_pixels=480000,
            file_size=102400
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["filename"] == "image.jpg"
        assert metadata_dict["size"] == (800, 600)
        assert metadata_dict["format"] == "JPEG"


class TestColorData:
    """Test ColorData model."""
    
    def test_color_data_creation(self, sample_color_data):
        """Test ColorData creation."""
        assert sample_color_data.unique_color_count == 4
        assert len(sample_color_data.colors) == 4
        assert sample_color_data.dominant_color.hex == "#ff0000"
    
    def test_color_data_to_dict(self, sample_color_data):
        """Test ColorData dictionary conversion."""
        data_dict = sample_color_data.to_dict()
        
        assert "image_metadata" in data_dict
        assert "colors" in data_dict
        assert "unique_color_count" in data_dict
        assert data_dict["unique_color_count"] == 4
    
    def test_color_data_to_json(self, sample_color_data):
        """Test ColorData JSON conversion."""
        json_str = sample_color_data.to_json()
        parsed = json.loads(json_str)
        
        assert "colors" in parsed
        assert len(parsed["colors"]) == 4
        assert parsed["unique_color_count"] == 4
    
    def test_color_data_save_json(self, sample_color_data, temp_dir):
        """Test ColorData JSON file saving."""
        output_path = temp_dir / "color_data.json"
        sample_color_data.save_json(output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "colors" in data
        assert len(data["colors"]) == 4


class TestColorCluster:
    """Test ColorCluster model."""
    
    def test_color_cluster_creation(self, sample_colors):
        """Test ColorCluster creation."""
        centroid = sample_colors[0]
        cluster_colors = sample_colors[:2]
        
        cluster = ColorCluster(
            cluster_id=0,
            centroid=centroid,
            colors=cluster_colors,
            size=2,
            variance=10.5
        )
        
        assert cluster.cluster_id == 0
        assert cluster.centroid == centroid
        assert len(cluster.colors) == 2
        assert cluster.size == 2
        assert cluster.variance == 10.5
    
    def test_color_cluster_to_dict(self, sample_colors):
        """Test ColorCluster dictionary conversion."""
        cluster = ColorCluster(
            cluster_id=0,
            centroid=sample_colors[0],
            colors=sample_colors[:2],
            size=2,
            variance=10.5
        )
        
        cluster_dict = cluster.to_dict()
        
        assert cluster_dict["cluster_id"] == 0
        assert "centroid" in cluster_dict
        assert len(cluster_dict["colors"]) == 2
        assert cluster_dict["size"] == 2


class TestColorHarmony:
    """Test ColorHarmony model."""
    
    def test_color_harmony_creation(self, sample_colors):
        """Test ColorHarmony creation."""
        harmony = ColorHarmony(
            harmony_type="complementary",
            base_color=sample_colors[0],
            harmony_colors=sample_colors[1:3],
            confidence=0.8
        )
        
        assert harmony.harmony_type == "complementary"
        assert harmony.base_color == sample_colors[0]
        assert len(harmony.harmony_colors) == 2
        assert harmony.confidence == 0.8
    
    def test_color_harmony_to_dict(self, sample_colors):
        """Test ColorHarmony dictionary conversion."""
        harmony = ColorHarmony(
            harmony_type="triadic",
            base_color=sample_colors[0],
            harmony_colors=sample_colors[1:4],
            confidence=0.9
        )
        
        harmony_dict = harmony.to_dict()
        
        assert harmony_dict["harmony_type"] == "triadic"
        assert "base_color" in harmony_dict
        assert len(harmony_dict["harmony_colors"]) == 3
        assert harmony_dict["confidence"] == 0.9


class TestPaletteData:
    """Test PaletteData model."""
    
    def test_palette_data_creation(self, sample_colors):
        """Test PaletteData creation."""
        palette = PaletteData(
            name="Test Palette",
            colors=sample_colors,
            palette_type="dominant",
            source_image="test.jpg"
        )
        
        assert palette.name == "Test Palette"
        assert len(palette.colors) == 4
        assert palette.palette_type == "dominant"
        assert palette.source_image == "test.jpg"
    
    def test_palette_data_to_dict(self, sample_colors):
        """Test PaletteData dictionary conversion."""
        palette = PaletteData(
            name="Test Palette",
            colors=sample_colors,
            palette_type="vibrant"
        )
        
        palette_dict = palette.to_dict()
        
        assert palette_dict["name"] == "Test Palette"
        assert len(palette_dict["colors"]) == 4
        assert palette_dict["palette_type"] == "vibrant"


class TestAnalysisResult:
    """Test AnalysisResult model."""
    
    def test_analysis_result_creation(self, sample_color_data, sample_colors):
        """Test AnalysisResult creation."""
        palettes = [PaletteData(
            name="Test Palette",
            colors=sample_colors,
            palette_type="dominant"
        )]
        
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.5,
            algorithm_used="kmeans",
            settings={"max_colors": 16}
        )
        
        assert result.color_data == sample_color_data
        assert len(result.palettes) == 1
        assert result.processing_time == 1.5
        assert result.algorithm_used == "kmeans"
        assert result.settings["max_colors"] == 16
    
    def test_analysis_result_to_dict(self, sample_color_data, sample_colors):
        """Test AnalysisResult dictionary conversion."""
        palettes = [PaletteData(
            name="Test Palette",
            colors=sample_colors,
            palette_type="dominant"
        )]
        
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.5,
            algorithm_used="kmeans",
            settings={}
        )
        
        result_dict = result.to_dict()
        
        assert "color_data" in result_dict
        assert "palettes" in result_dict
        assert result_dict["processing_time"] == 1.5
        assert result_dict["algorithm_used"] == "kmeans"
    
    def test_analysis_result_to_json(self, sample_color_data, sample_colors):
        """Test AnalysisResult JSON conversion."""
        palettes = [PaletteData(
            name="Test Palette",
            colors=sample_colors,
            palette_type="dominant"
        )]
        
        result = AnalysisResult(
            color_data=sample_color_data,
            palettes=palettes,
            processing_time=1.5,
            algorithm_used="kmeans",
            settings={}
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert "color_data" in parsed
        assert "palettes" in parsed
        assert parsed["algorithm_used"] == "kmeans"


class TestExportFormat:
    """Test ExportFormat enum."""
    
    def test_export_format_values(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSS.value == "css"
        assert ExportFormat.SCSS.value == "scss"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.PDF.value == "pdf"
    
    def test_export_format_iteration(self):
        """Test ExportFormat enum iteration."""
        formats = list(ExportFormat)
        
        assert ExportFormat.JSON in formats
        assert ExportFormat.CSS in formats
        assert len(formats) > 10  # We have many export formats
