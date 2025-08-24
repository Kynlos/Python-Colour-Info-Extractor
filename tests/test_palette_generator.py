"""Tests for palette generation functionality."""

import pytest
import math
from PIL import Image

from pycolour_extract.core.palette_generator import PaletteGenerator
from pycolour_extract.models.color_data import ColorInfo, PaletteData


class TestPaletteGenerator:
    """Test PaletteGenerator class."""
    
    def test_palette_generator_initialization(self):
        """Test PaletteGenerator initialization."""
        generator = PaletteGenerator()
        assert generator.golden_ratio == pytest.approx((1 + math.sqrt(5)) / 2)
    
    def test_generate_palette_variations_dominant(self, sample_colors):
        """Test dominant palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "dominant", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.name is not None
            assert len(palette.colors) > 0
            assert palette.palette_type is not None
    
    def test_generate_palette_variations_harmony(self, sample_colors):
        """Test harmony palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "harmony", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert "harmony" in palette.palette_type.lower() or "complementary" in palette.palette_type.lower()
    
    def test_generate_palette_variations_monochromatic(self, sample_colors):
        """Test monochromatic palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "monochromatic", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert "monochromatic" in palette.palette_type.lower()
            
            # Check that colors have similar hues
            if len(palette.colors) > 1:
                base_hue = palette.colors[0].hsv[0] if palette.colors[0].hsv else 0
                for color in palette.colors[1:]:
                    if color.hsv:
                        hue_diff = abs(color.hsv[0] - base_hue)
                        # Allow for some variation and hue wrap-around
                        assert hue_diff < 0.2 or hue_diff > 0.8
    
    def test_generate_palette_variations_analogous(self, sample_colors):
        """Test analogous palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "analogous", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "analogous"
    
    def test_generate_palette_variations_complementary(self, sample_colors):
        """Test complementary palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "complementary", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
    
    def test_generate_palette_variations_triadic(self, sample_colors):
        """Test triadic palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "triadic", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "triadic"
            
            # Should have colors approximately 120 degrees apart
            if len(palette.colors) >= 3:
                hues = [c.hsv[0] * 360 for c in palette.colors[:3] if c.hsv]
                if len(hues) >= 3:
                    # Sort hues for comparison
                    hues.sort()
                    # Check if roughly 120 degrees apart (allow some tolerance)
                    diff1 = hues[1] - hues[0]
                    diff2 = hues[2] - hues[1]
                    # Due to generation variations, just check they're reasonably spaced
                    assert 60 <= diff1 <= 180
                    assert 60 <= diff2 <= 180
    
    def test_generate_palette_variations_vibrant(self, sample_colors):
        """Test vibrant palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "vibrant", 5)
        
        # May return empty if no vibrant colors can be generated
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "vibrant"
            
            # Vibrant colors should have high saturation
            for color in palette.colors:
                if color.hsv:
                    assert color.hsv[1] > 0.5  # High saturation
    
    def test_generate_palette_variations_muted(self, sample_colors):
        """Test muted palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "muted", 5)
        
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "muted"
            
            # Muted colors should have lower saturation
            for color in palette.colors:
                if color.hsv:
                    assert color.hsv[1] < 0.6  # Lower saturation
    
    def test_generate_palette_variations_pastel(self, sample_colors):
        """Test pastel palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "pastel", 5)
        
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "pastel"
            
            # Pastel colors should have low saturation and high value
            for color in palette.colors:
                if color.hsv:
                    assert color.hsv[1] <= 0.4  # Low saturation
                    assert color.hsv[2] >= 0.7  # High value (brightness)
    
    def test_generate_palette_variations_material(self, sample_colors):
        """Test Material Design palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "material", 5)
        
        assert len(palettes) > 0
        for palette in palettes:
            assert isinstance(palette, PaletteData)
            assert palette.palette_type == "material"
    
    def test_generate_palette_variations_gradient(self, sample_colors):
        """Test gradient palette generation."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations(sample_colors, "gradient", 5)
        
        if len(sample_colors) >= 2:  # Need at least 2 colors for gradient
            assert len(palettes) > 0
            for palette in palettes:
                assert isinstance(palette, PaletteData)
                assert palette.palette_type == "gradient"
    
    def test_generate_palette_variations_empty_colors(self):
        """Test palette generation with empty color list."""
        generator = PaletteGenerator()
        palettes = generator.generate_palette_variations([], "dominant", 5)
        
        assert len(palettes) == 0
    
    def test_create_palette_image_swatches(self, sample_colors):
        """Test swatch-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "swatches", 400, 100, True)
        
        assert isinstance(image, Image.Image)
        assert image.size == (400, 100)
        assert image.mode == "RGB"
    
    def test_create_palette_image_gradient(self, sample_colors):
        """Test gradient-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "gradient", 400, 100)
        
        assert isinstance(image, Image.Image)
        assert image.size == (400, 100)
        assert image.mode == "RGB"
    
    def test_create_palette_image_circle(self, sample_colors):
        """Test circle-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "circle", 200, 200, True)
        
        assert isinstance(image, Image.Image)
        assert image.size == (200, 200)
        assert image.mode == "RGB"
    
    def test_create_palette_image_hexagon(self, sample_colors):
        """Test hexagon-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "hexagon", 200, 200)
        
        assert isinstance(image, Image.Image)
        assert image.size == (200, 200)
        assert image.mode == "RGB"
    
    def test_create_palette_image_squares(self, sample_colors):
        """Test squares-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "squares", 200, 200, True)
        
        assert isinstance(image, Image.Image)
        assert image.size == (200, 200)
        assert image.mode == "RGB"
    
    def test_create_palette_image_waves(self, sample_colors):
        """Test waves-style palette image creation."""
        generator = PaletteGenerator()
        image = generator.create_palette_image(sample_colors, "waves", 400, 100)
        
        assert isinstance(image, Image.Image)
        assert image.size == (400, 100)
        assert image.mode == "RGB"
    
    def test_create_palette_image_empty_colors(self):
        """Test palette image creation with empty color list."""
        generator = PaletteGenerator()
        image = generator.create_palette_image([], "swatches", 400, 100)
        
        assert isinstance(image, Image.Image)
        assert image.size == (400, 100)
        # Should be white background
        assert image.getpixel((0, 0)) == (255, 255, 255)
    
    def test_create_palette_image_single_color(self):
        """Test palette image creation with single color."""
        single_color = [ColorInfo(rgb=(255, 0, 0), hex="#ff0000")]
        
        generator = PaletteGenerator()
        image = generator.create_palette_image(single_color, "gradient", 200, 100)
        
        assert isinstance(image, Image.Image)
        assert image.size == (200, 100)
        # Should be solid red
        assert image.getpixel((100, 50)) == (255, 0, 0)
    
    def test_analyze_palette_mood(self, sample_colors):
        """Test palette mood analysis."""
        generator = PaletteGenerator()
        mood_analysis = generator.analyze_palette_mood(sample_colors)
        
        assert isinstance(mood_analysis, dict)
        assert "overall_mood" in mood_analysis
        assert "temperature" in mood_analysis
        assert "energy_level" in mood_analysis
        assert "sophistication" in mood_analysis
        assert "harmony_score" in mood_analysis
        assert "contrast_level" in mood_analysis
        assert "color_balance" in mood_analysis
        assert "seasonal_association" in mood_analysis
        assert "style_recommendations" in mood_analysis
        
        # Check value ranges
        assert 0 <= mood_analysis["energy_level"] <= 1
        assert 0 <= mood_analysis["sophistication"] <= 1
        assert 0 <= mood_analysis["harmony_score"] <= 1
        assert 0 <= mood_analysis["contrast_level"]
        
        # Check that style recommendations is a list
        assert isinstance(mood_analysis["style_recommendations"], list)
    
    def test_analyze_palette_mood_empty_colors(self):
        """Test mood analysis with empty colors."""
        generator = PaletteGenerator()
        mood_analysis = generator.analyze_palette_mood([])
        
        assert "error" in mood_analysis
    
    def test_suggest_palette_improvements(self, sample_colors):
        """Test palette improvement suggestions."""
        generator = PaletteGenerator()
        suggestions = generator.suggest_palette_improvements(sample_colors)
        
        assert isinstance(suggestions, list)
        
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "type" in suggestion
            assert "priority" in suggestion
            assert "message" in suggestion
            assert "solution" in suggestion
            
            # Check priority values
            assert suggestion["priority"] in ["high", "medium", "low"]
    
    def test_suggest_palette_improvements_empty_colors(self):
        """Test improvement suggestions with empty colors."""
        generator = PaletteGenerator()
        suggestions = generator.suggest_palette_improvements([])
        
        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == "error"
    
    def test_determine_overall_mood_variations(self):
        """Test overall mood determination with different color combinations."""
        generator = PaletteGenerator()
        
        # Test dark colors
        dark_colors = [
            ColorInfo(rgb=(50, 50, 50), hex="#323232", hsv=(0, 0, 0.2)),
            ColorInfo(rgb=(30, 30, 30), hex="#1e1e1e", hsv=(0, 0, 0.12))
        ]
        mood = generator._determine_overall_mood(dark_colors)
        assert mood in ["sophisticated", "cozy", "neutral"]
        
        # Test bright colors
        bright_colors = [
            ColorInfo(rgb=(255, 255, 0), hex="#ffff00", hsv=(0.17, 1.0, 1.0)),
            ColorInfo(rgb=(255, 0, 255), hex="#ff00ff", hsv=(0.83, 1.0, 1.0))
        ]
        mood = generator._determine_overall_mood(bright_colors)
        assert mood in ["energetic", "vibrant", "fresh", "cheerful"]
    
    def test_analyze_temperature_variations(self):
        """Test temperature analysis with different color combinations."""
        generator = PaletteGenerator()
        
        # Test warm colors
        warm_colors = [
            ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0), percentage=50.0),
            ColorInfo(rgb=(255, 165, 0), hex="#ffa500", hsv=(0.11, 1.0, 1.0), percentage=50.0)
        ]
        temp_analysis = generator._analyze_temperature_relationships(warm_colors)
        assert temp_analysis["temperature_balance"] == "warm"
        
        # Test cool colors
        cool_colors = [
            ColorInfo(rgb=(0, 0, 255), hex="#0000ff", hsv=(0.67, 1.0, 1.0), percentage=50.0),
            ColorInfo(rgb=(0, 255, 255), hex="#00ffff", hsv=(0.5, 1.0, 1.0), percentage=50.0)
        ]
        temp_analysis = generator._analyze_temperature_relationships(cool_colors)
        assert temp_analysis["temperature_balance"] == "cool"
    
    def test_calculate_harmony_score_variations(self):
        """Test harmony score calculation with different color relationships."""
        generator = PaletteGenerator()
        
        # Test complementary colors (should have good harmony)
        complementary = [
            ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0)),
            ColorInfo(rgb=(0, 255, 255), hex="#00ffff", hsv=(0.5, 1.0, 1.0))
        ]
        harmony_score = generator._calculate_harmony_score(complementary)
        assert 0.5 <= harmony_score <= 1.0  # Should be reasonably harmonious
        
        # Test single color (perfect harmony)
        single_color = [ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0))]
        harmony_score = generator._calculate_harmony_score(single_color)
        assert harmony_score == 1.0
    
    def test_seasonal_association_detection(self):
        """Test seasonal association detection."""
        generator = PaletteGenerator()
        
        # Test spring colors (greens, pastels)
        spring_colors = [
            ColorInfo(rgb=(144, 238, 144), hex="#90ee90", hsv=(0.33, 0.39, 0.93), percentage=100.0)
        ]
        seasonal = generator._determine_seasonal_association(spring_colors)
        assert seasonal in ["spring", "summer"]  # Light green could be either
        
        # Test autumn colors (oranges, reds)
        autumn_colors = [
            ColorInfo(rgb=(255, 140, 0), hex="#ff8c00", hsv=(0.09, 1.0, 1.0), percentage=100.0)
        ]
        seasonal = generator._determine_seasonal_association(autumn_colors)
        assert seasonal == "autumn"
    
    def test_style_recommendations_generation(self):
        """Test style recommendations based on color characteristics."""
        generator = PaletteGenerator()
        
        # Test high saturation colors
        vibrant_colors = [
            ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0)),
            ColorInfo(rgb=(0, 255, 0), hex="#00ff00", hsv=(0.33, 1.0, 1.0))
        ]
        recommendations = generator._get_style_recommendations(vibrant_colors)
        assert any("modern" in rec.lower() or "energetic" in rec.lower() for rec in recommendations)
        
        # Test low saturation colors
        muted_colors = [
            ColorInfo(rgb=(128, 128, 128), hex="#808080", hsv=(0, 0, 0.5)),
            ColorInfo(rgb=(160, 160, 160), hex="#a0a0a0", hsv=(0, 0, 0.63))
        ]
        recommendations = generator._get_style_recommendations(muted_colors)
        assert any("professional" in rec.lower() or "sophisticated" in rec.lower() for rec in recommendations)
    
    def test_color_blindness_accessibility_checks(self, sample_colors):
        """Test color blindness accessibility checking."""
        generator = PaletteGenerator()
        suggestions = generator.suggest_palette_improvements(sample_colors)
        
        # Look for accessibility-related suggestions
        accessibility_suggestions = [s for s in suggestions if s["type"] == "accessibility"]
        
        # Should provide some accessibility feedback
        # (May be empty if the palette is already accessible)
        for suggestion in accessibility_suggestions:
            assert "accessibility" in suggestion["message"].lower()
    
    def test_create_color_info_hsv_utility(self):
        """Test HSV to ColorInfo utility method."""
        generator = PaletteGenerator()
        
        # Test red color
        color = generator._create_color_info_hsv(0.0, 1.0, 1.0)
        assert color.rgb == (255, 0, 0)
        assert color.hex == "#ff0000"
        assert color.hsv == (0.0, 1.0, 1.0)
        
        # Test blue color
        color = generator._create_color_info_hsv(0.67, 1.0, 1.0)
        # Allow for some rounding in RGB conversion
        assert abs(color.rgb[2] - 255) <= 1  # Should be mostly blue
        assert color.rgb[0] == 0  # No red
        assert color.rgb[1] == 0  # No green
    
    def test_palette_generation_with_various_counts(self, sample_colors):
        """Test palette generation with different color counts."""
        generator = PaletteGenerator()
        
        for count in [3, 5, 8, 12]:
            palettes = generator.generate_palette_variations(sample_colors, "dominant", count)
            
            for palette in palettes:
                # Should not exceed requested count (may be less if not enough source colors)
                assert len(palette.colors) <= count
    
    def test_all_palette_styles(self, sample_colors):
        """Test all supported palette styles."""
        generator = PaletteGenerator()
        
        styles = [
            "dominant", "harmony", "monochromatic", "analogous", "complementary",
            "split_complementary", "triadic", "tetradic", "vibrant", "muted",
            "pastel", "dark", "light", "gradient", "material", "web_safe"
        ]
        
        for style in styles:
            palettes = generator.generate_palette_variations(sample_colors, style, 5)
            # Some styles might return empty lists if conditions aren't met
            for palette in palettes:
                assert isinstance(palette, PaletteData)
                assert palette.name is not None
                assert len(palette.colors) > 0
