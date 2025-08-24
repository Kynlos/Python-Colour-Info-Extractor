"""Tests for color extraction functionality."""

import pytest
import numpy as np
from PIL import Image

from pycolour_extract.core.color_extractor import ColorExtractor
from pycolour_extract.models.color_data import ColorInfo, AnalysisResult


class TestColorExtractor:
    """Test ColorExtractor class."""
    
    def test_color_extractor_initialization(self):
        """Test ColorExtractor initialization."""
        extractor = ColorExtractor()
        assert extractor.algorithm == "kmeans"
        assert extractor.max_colors == 256
        
        extractor_custom = ColorExtractor(algorithm="dbscan", max_colors=50)
        assert extractor_custom.algorithm == "dbscan"
        assert extractor_custom.max_colors == 50
    
    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            ColorExtractor(algorithm="invalid_algorithm")
    
    def test_supported_algorithms(self):
        """Test all supported algorithms can be initialized."""
        algorithms = ["kmeans", "dbscan", "median_cut", "octree", "histogram"]
        
        for algorithm in algorithms:
            extractor = ColorExtractor(algorithm=algorithm)
            assert extractor.algorithm == algorithm
    
    @pytest.mark.parametrize("algorithm", ["kmeans", "median_cut", "octree"])
    def test_extract_colors_basic(self, sample_image, algorithm):
        """Test basic color extraction with different algorithms."""
        extractor = ColorExtractor(algorithm=algorithm, max_colors=8)
        result = extractor.extract_colors(sample_image)
        
        assert isinstance(result, AnalysisResult)
        assert result.color_data is not None
        assert len(result.color_data.colors) > 0
        assert result.color_data.unique_color_count > 0
        assert result.algorithm_used == algorithm
        assert result.processing_time > 0
    
    def test_extract_colors_kmeans(self, sample_image):
        """Test K-means color extraction specifically."""
        extractor = ColorExtractor(algorithm="kmeans", max_colors=4)
        result = extractor.extract_colors(sample_image, n_clusters=4)
        
        # Should find approximately 4 colors (our quadrant colors)
        assert len(result.color_data.colors) <= 4
        assert result.color_data.dominant_color is not None
        assert result.color_data.average_color is not None
        
        # Check that colors have proper attributes
        for color in result.color_data.colors:
            assert isinstance(color, ColorInfo)
            assert len(color.rgb) == 3
            assert color.hex.startswith('#')
            assert color.frequency >= 0
            assert color.percentage >= 0
    
    def test_extract_colors_dbscan(self, sample_image):
        """Test DBSCAN color extraction."""
        extractor = ColorExtractor(algorithm="dbscan", max_colors=10)
        result = extractor.extract_colors(sample_image, eps=10, min_samples=50)
        
        assert len(result.color_data.colors) > 0
        assert all(color.frequency > 0 for color in result.color_data.colors)
    
    def test_extract_colors_histogram(self, gradient_image):
        """Test histogram-based extraction."""
        extractor = ColorExtractor(algorithm="histogram", max_colors=20)
        result = extractor.extract_colors(gradient_image, bins=16)
        
        assert len(result.color_data.colors) > 0
        # Gradient should produce multiple colors
        assert len(result.color_data.colors) >= 2
    
    def test_extract_metadata(self, sample_image):
        """Test image metadata extraction."""
        extractor = ColorExtractor()
        result = extractor.extract_colors(sample_image)
        
        metadata = result.color_data.image_metadata
        assert metadata.filename == "test_image.png"
        assert metadata.size == (200, 200)
        assert metadata.format == "PNG"
        assert metadata.mode == "RGB"
        assert metadata.total_pixels == 40000
        assert metadata.file_size > 0
    
    def test_color_enhancement(self, sample_image):
        """Test color enhancement with additional color spaces."""
        extractor = ColorExtractor(max_colors=4)
        result = extractor.extract_colors(sample_image)
        
        for color in result.color_data.colors:
            # Check RGB
            assert len(color.rgb) == 3
            assert all(0 <= c <= 255 for c in color.rgb)
            
            # Check hex
            assert color.hex.startswith('#')
            assert len(color.hex) == 7
            
            # Check HSV conversion
            if color.hsv:
                h, s, v = color.hsv
                assert 0 <= h <= 1
                assert 0 <= s <= 1
                assert 0 <= v <= 1
            
            # Check LAB conversion
            if color.lab:
                l, a, b = color.lab
                assert 0 <= l <= 100  # L is 0-100
            
            # Check luminance
            if color.luminance is not None:
                assert 0 <= color.luminance <= 1
    
    def test_clustering_analysis(self, sample_image):
        """Test color clustering functionality."""
        extractor = ColorExtractor(max_colors=8)
        result = extractor.extract_colors(sample_image, n_clusters=3)
        
        if result.color_data.clusters:
            assert len(result.color_data.clusters) <= 3
            
            for cluster in result.color_data.clusters:
                assert cluster.cluster_id >= 0
                assert cluster.centroid is not None
                assert len(cluster.colors) > 0
                assert cluster.size == len(cluster.colors)
                assert cluster.variance >= 0
    
    def test_harmony_detection(self, sample_image):
        """Test color harmony detection."""
        extractor = ColorExtractor(max_colors=8)
        result = extractor.extract_colors(sample_image)
        
        if result.color_data.harmonies:
            for harmony in result.color_data.harmonies:
                assert harmony.harmony_type in [
                    "complementary", "triadic", "analogous"
                ]
                assert harmony.base_color is not None
                assert len(harmony.harmony_colors) > 0
                assert 0 <= harmony.confidence <= 1
    
    def test_accessibility_analysis(self, sample_image):
        """Test accessibility analysis."""
        extractor = ColorExtractor(max_colors=4)
        result = extractor.extract_colors(sample_image)
        
        if result.color_data.accessibility_score is not None:
            assert 0 <= result.color_data.accessibility_score <= 1
    
    def test_color_temperature_calculation(self, sample_image):
        """Test color temperature calculation."""
        extractor = ColorExtractor(max_colors=4)
        result = extractor.extract_colors(sample_image)
        
        if result.color_data.color_temperature is not None:
            # Color temperature should be in reasonable range (1000K - 25000K)
            assert 1000 <= result.color_data.color_temperature <= 25000
    
    def test_vibrance_calculation(self, sample_image):
        """Test vibrance calculation."""
        extractor = ColorExtractor(max_colors=4)
        result = extractor.extract_colors(sample_image)
        
        if result.color_data.vibrance is not None:
            assert 0 <= result.color_data.vibrance <= 1
    
    def test_saturation_distribution(self, sample_image):
        """Test saturation distribution analysis."""
        extractor = ColorExtractor(max_colors=4)
        result = extractor.extract_colors(sample_image)
        
        if result.color_data.saturation_distribution:
            dist = result.color_data.saturation_distribution
            assert "low_saturation" in dist
            assert "medium_saturation" in dist
            assert "high_saturation" in dist
            
            # Percentages should sum to approximately 100
            total = sum(dist.values())
            assert 90 <= total <= 110  # Allow some rounding error
    
    def test_palette_generation(self, sample_image):
        """Test palette generation."""
        extractor = ColorExtractor(max_colors=8)
        result = extractor.extract_colors(sample_image)
        
        assert len(result.palettes) > 0
        
        for palette in result.palettes:
            assert palette.name is not None
            assert len(palette.colors) > 0
            assert palette.palette_type is not None
            assert palette.source_image is not None
    
    def test_monochrome_image(self, monochrome_image):
        """Test extraction from monochrome image."""
        extractor = ColorExtractor(max_colors=5)
        result = extractor.extract_colors(monochrome_image)
        
        # Should still work with monochrome images
        assert len(result.color_data.colors) >= 1
        assert result.color_data.dominant_color is not None
    
    def test_complex_image(self, complex_image):
        """Test extraction from complex multi-color image."""
        extractor = ColorExtractor(algorithm="kmeans", max_colors=16)
        result = extractor.extract_colors(complex_image)
        
        # Complex image should yield multiple colors
        assert len(result.color_data.colors) > 5
        assert result.color_data.unique_color_count > 5
    
    def test_utility_methods(self):
        """Test utility methods."""
        extractor = ColorExtractor()
        
        # Test RGB to hex conversion
        hex_color = extractor._rgb_to_hex((255, 0, 0))
        assert hex_color == "#ff0000"
        
        # Test RGB to CMYK conversion
        cmyk = extractor._rgb_to_cmyk(255, 0, 0)
        assert len(cmyk) == 4
        assert all(0 <= c <= 1 for c in cmyk)
        
        # Test luminance calculation
        luminance = extractor._calculate_luminance(255, 255, 255)
        assert luminance == pytest.approx(1.0, rel=1e-2)
        
        luminance_black = extractor._calculate_luminance(0, 0, 0)
        assert luminance_black == pytest.approx(0.0, abs=1e-3)
        
        # Test contrast ratio calculation
        contrast = extractor._calculate_contrast_ratio((255, 255, 255), (0, 0, 0))
        assert contrast > 20  # Should be high contrast
    
    def test_error_handling(self, temp_dir):
        """Test error handling for invalid inputs."""
        extractor = ColorExtractor()
        
        # Test with non-existent file
        non_existent = temp_dir / "non_existent.jpg"
        with pytest.raises(FileNotFoundError):
            extractor.extract_colors(non_existent)
    
    def test_different_image_formats(self, temp_dir):
        """Test extraction from different image formats."""
        # Create images in different formats
        base_image = Image.new('RGB', (50, 50), color=(255, 0, 0))
        
        formats = [('JPEG', 'jpg'), ('PNG', 'png'), ('BMP', 'bmp')]
        
        for format_name, extension in formats:
            image_path = temp_dir / f"test_image.{extension}"
            if format_name == 'JPEG':
                base_image.save(image_path, format=format_name, quality=95)
            else:
                base_image.save(image_path, format=format_name)
            
            extractor = ColorExtractor(max_colors=3)
            result = extractor.extract_colors(image_path)
            
            assert len(result.color_data.colors) > 0
            assert result.color_data.image_metadata.format == format_name
    
    def test_max_colors_limit(self, complex_image):
        """Test that max_colors parameter is respected."""
        max_colors = 5
        extractor = ColorExtractor(algorithm="kmeans", max_colors=max_colors)
        result = extractor.extract_colors(complex_image)
        
        assert len(result.color_data.colors) <= max_colors
    
    def test_processing_time_recorded(self, sample_image):
        """Test that processing time is recorded."""
        extractor = ColorExtractor()
        result = extractor.extract_colors(sample_image)
        
        assert result.processing_time > 0
        assert result.processing_time < 60  # Should be reasonable
    
    def test_settings_recorded(self, sample_image):
        """Test that extraction settings are recorded."""
        extractor = ColorExtractor(algorithm="kmeans", max_colors=10)
        result = extractor.extract_colors(sample_image, n_clusters=3)
        
        assert "n_clusters" in result.settings
        assert result.settings["n_clusters"] == 3
