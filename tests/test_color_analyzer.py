"""Tests for color analysis functionality."""

import pytest
import math
from pycolour_extract.core.color_analyzer import ColorAnalyzer
from pycolour_extract.models.color_data import ColorInfo


class TestColorAnalyzer:
    """Test ColorAnalyzer class."""
    
    def test_color_analyzer_initialization(self):
        """Test ColorAnalyzer initialization."""
        analyzer = ColorAnalyzer()
        assert analyzer.golden_ratio == pytest.approx((1 + math.sqrt(5)) / 2)
    
    def test_analyze_color_relationships(self, sample_colors):
        """Test color relationship analysis."""
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(sample_colors)
        
        assert isinstance(relationships, dict)
        assert "harmony_types" in relationships
        assert "color_distances" in relationships
        assert "perceptual_groupings" in relationships
        assert "complementary_pairs" in relationships
        assert "analogous_groups" in relationships
        assert "temperature_analysis" in relationships
        assert "contrast_matrix" in relationships
        assert "accessibility_pairs" in relationships
    
    def test_analyze_color_relationships_insufficient_colors(self):
        """Test relationship analysis with insufficient colors."""
        analyzer = ColorAnalyzer()
        single_color = [ColorInfo(rgb=(255, 0, 0), hex="#ff0000")]
        
        relationships = analyzer.analyze_color_relationships(single_color)
        assert "error" in relationships
    
    def test_calculate_color_emotion(self, sample_colors):
        """Test color emotion calculation."""
        analyzer = ColorAnalyzer()
        emotions = analyzer.calculate_color_emotion(sample_colors)
        
        assert isinstance(emotions, dict)
        
        expected_emotions = [
            "energy", "calmness", "warmth", "sophistication", 
            "playfulness", "trustworthiness", "creativity", "luxury"
        ]
        
        for emotion in expected_emotions:
            assert emotion in emotions
            assert 0 <= emotions[emotion] <= 1
    
    def test_calculate_color_emotion_empty_list(self):
        """Test emotion calculation with empty color list."""
        analyzer = ColorAnalyzer()
        emotions = analyzer.calculate_color_emotion([])
        
        for emotion_value in emotions.values():
            assert emotion_value == 0.0
    
    def test_analyze_brand_personality(self, sample_colors):
        """Test brand personality analysis."""
        analyzer = ColorAnalyzer()
        personality = analyzer.analyze_brand_personality(sample_colors)
        
        assert isinstance(personality, dict)
        assert "personality_scores" in personality
        assert "dominant_traits" in personality
        assert "overall_personality" in personality
        
        # Check personality scores
        scores = personality["personality_scores"]
        expected_traits = [
            "modern", "traditional", "professional", "casual", "bold", 
            "subtle", "masculine", "feminine", "youthful", "mature"
        ]
        
        for trait in expected_traits:
            assert trait in scores
            assert 0 <= scores[trait] <= 1
        
        # Check dominant traits
        dominant = personality["dominant_traits"]
        assert isinstance(dominant, list)
        assert len(dominant) == 3
        
        for trait_info in dominant:
            assert "trait" in trait_info
            assert "score" in trait_info
            assert 0 <= trait_info["score"] <= 1
    
    def test_analyze_brand_personality_empty_list(self):
        """Test brand personality analysis with empty color list."""
        analyzer = ColorAnalyzer()
        personality = analyzer.analyze_brand_personality([])
        
        assert "error" in personality
    
    def test_generate_color_suggestions_harmony(self, sample_colors):
        """Test color suggestion generation - harmony type."""
        analyzer = ColorAnalyzer()
        suggestions = analyzer.generate_color_suggestions(sample_colors, "harmony")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert len(suggestions) <= 8
        
        for suggestion in suggestions:
            assert isinstance(suggestion, ColorInfo)
            assert len(suggestion.rgb) == 3
            assert suggestion.hex.startswith('#')
    
    def test_generate_color_suggestions_monochromatic(self, sample_colors):
        """Test color suggestion generation - monochromatic type."""
        analyzer = ColorAnalyzer()
        suggestions = analyzer.generate_color_suggestions(sample_colors, "monochromatic")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # All suggestions should have similar hue to base color
        base_hue = sample_colors[0].hsv[0] if sample_colors[0].hsv else 0
        
        for suggestion in suggestions:
            if suggestion.hsv:
                hue_diff = abs(suggestion.hsv[0] - base_hue)
                # Allow for some variation in monochromatic scheme
                assert hue_diff < 0.2 or hue_diff > 0.8  # Account for hue wrap-around
    
    def test_generate_color_suggestions_analogous(self, sample_colors):
        """Test color suggestion generation - analogous type."""
        analyzer = ColorAnalyzer()
        suggestions = analyzer.generate_color_suggestions(sample_colors, "analogous")
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
    
    def test_generate_color_suggestions_accessibility(self, sample_colors):
        """Test color suggestion generation - accessibility type."""
        analyzer = ColorAnalyzer()
        suggestions = analyzer.generate_color_suggestions(sample_colors, "accessibility")
        
        assert isinstance(suggestions, list)
        # May return empty list if no good accessible colors can be generated
        
        for suggestion in suggestions:
            assert isinstance(suggestion, ColorInfo)
    
    def test_generate_color_suggestions_empty_list(self):
        """Test color suggestions with empty color list."""
        analyzer = ColorAnalyzer()
        suggestions = analyzer.generate_color_suggestions([], "harmony")
        
        assert suggestions == []
    
    def test_calculate_color_blindness_impact(self, sample_colors):
        """Test color blindness impact calculation."""
        analyzer = ColorAnalyzer()
        impact = analyzer.calculate_color_blindness_impact(sample_colors)
        
        assert isinstance(impact, dict)
        assert "simulations" in impact
        assert "distinguishability_scores" in impact
        assert "overall_accessibility" in impact
        assert "recommendations" in impact
        
        # Check simulations
        simulations = impact["simulations"]
        expected_types = [
            "protanopia", "deuteranopia", "tritanopia",
            "protanomaly", "deuteranomaly", "tritanomaly"
        ]
        
        for cb_type in expected_types:
            assert cb_type in simulations
            assert len(simulations[cb_type]) == len(sample_colors)
            
            for simulated_color in simulations[cb_type]:
                assert len(simulated_color) == 3  # RGB tuple
                assert all(0 <= c <= 255 for c in simulated_color)
        
        # Check distinguishability scores
        scores = impact["distinguishability_scores"]
        for cb_type in expected_types:
            assert cb_type in scores
            assert 0 <= scores[cb_type] <= 1
        
        # Check overall accessibility
        assert 0 <= impact["overall_accessibility"] <= 1
        
        # Check recommendations
        assert isinstance(impact["recommendations"], list)
    
    def test_perceptual_grouping(self, sample_colors):
        """Test perceptual color grouping."""
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(sample_colors)
        
        groupings = relationships["perceptual_groupings"]
        assert isinstance(groupings, dict)
        
        # Should have some color groups
        total_grouped_colors = sum(len(colors) for colors in groupings.values())
        assert total_grouped_colors > 0
    
    def test_complementary_pair_detection(self):
        """Test complementary color pair detection."""
        # Create complementary colors (red and cyan)
        red = ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0))
        cyan = ColorInfo(rgb=(0, 255, 255), hex="#00ffff", hsv=(0.5, 1.0, 1.0))
        colors = [red, cyan]
        
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(colors)
        
        pairs = relationships["complementary_pairs"]
        assert len(pairs) > 0
        
        # Should detect the complementary pair
        found_complementary = any(
            (pair[0] == red and pair[1] == cyan) or 
            (pair[0] == cyan and pair[1] == red) 
            for pair in pairs
        )
        assert found_complementary
    
    def test_analogous_group_detection(self):
        """Test analogous color group detection."""
        # Create analogous colors (reds and oranges)
        red = ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0))
        orange = ColorInfo(rgb=(255, 128, 0), hex="#ff8000", hsv=(0.08, 1.0, 1.0))  # ~30 degrees
        red_orange = ColorInfo(rgb=(255, 64, 0), hex="#ff4000", hsv=(0.04, 1.0, 1.0))  # ~15 degrees
        
        colors = [red, orange, red_orange]
        
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(colors)
        
        groups = relationships["analogous_groups"]
        assert len(groups) > 0
        
        # Should detect at least one analogous group
        assert any(len(group) >= 2 for group in groups)
    
    def test_temperature_analysis(self, sample_colors):
        """Test color temperature analysis."""
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(sample_colors)
        
        temp_analysis = relationships["temperature_analysis"]
        assert isinstance(temp_analysis, dict)
        assert "warm_colors" in temp_analysis
        assert "cool_colors" in temp_analysis
        assert "neutral_colors" in temp_analysis
        assert "temperature_balance" in temp_analysis
        
        # Percentages should sum to approximately 100
        total_percentage = (
            temp_analysis["warm_percentage"] +
            temp_analysis["cool_percentage"] +
            temp_analysis["neutral_percentage"]
        )
        assert 90 <= total_percentage <= 110  # Allow for rounding
    
    def test_contrast_matrix(self, sample_colors):
        """Test contrast ratio matrix calculation."""
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(sample_colors)
        
        matrix = relationships["contrast_matrix"]
        assert isinstance(matrix, list)
        assert len(matrix) == len(sample_colors)
        
        for row in matrix:
            assert len(row) == len(sample_colors)
            for contrast_ratio in row:
                assert contrast_ratio >= 1.0  # Contrast ratios are always >= 1
    
    def test_accessibility_pairs(self, sample_colors):
        """Test accessibility pair detection."""
        analyzer = ColorAnalyzer()
        relationships = analyzer.analyze_color_relationships(sample_colors)
        
        accessible_pairs = relationships["accessibility_pairs"]
        assert isinstance(accessible_pairs, list)
        
        for pair in accessible_pairs:
            assert len(pair) == 3  # (color1, color2, contrast_ratio)
            color1, color2, contrast_ratio = pair
            assert isinstance(color1, ColorInfo)
            assert isinstance(color2, ColorInfo)
            assert contrast_ratio >= 4.5  # WCAG AA standard
    
    def test_harmony_type_detection(self):
        """Test specific color harmony detection."""
        analyzer = ColorAnalyzer()
        
        # Test monochromatic detection
        red1 = ColorInfo(rgb=(255, 0, 0), hex="#ff0000", hsv=(0.0, 1.0, 1.0))
        red2 = ColorInfo(rgb=(200, 0, 0), hex="#c80000", hsv=(0.0, 1.0, 0.8))
        red3 = ColorInfo(rgb=(150, 0, 0), hex="#960000", hsv=(0.0, 1.0, 0.6))
        
        mono_colors = [(0.0, 1.0, 1.0), (0.0, 1.0, 0.8), (0.0, 1.0, 0.6)]
        assert analyzer._is_monochromatic(mono_colors)
        
        # Test complementary detection
        comp_colors = [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0)]  # Red and Cyan
        assert analyzer._is_complementary(comp_colors)
        
        # Test triadic detection
        triadic_colors = [(0.0, 1.0, 1.0), (120.0, 1.0, 1.0), (240.0, 1.0, 1.0)]
        assert analyzer._is_triadic(triadic_colors)
    
    def test_color_blindness_simulation_methods(self):
        """Test individual color blindness simulation methods."""
        analyzer = ColorAnalyzer()
        
        # Test with red color
        red_rgb = (1.0, 0.0, 0.0)  # Normalized RGB
        
        # Test each simulation method
        protanopia = analyzer._simulate_protanopia(*red_rgb)
        deuteranopia = analyzer._simulate_deuteranopia(*red_rgb)
        tritanopia = analyzer._simulate_tritanopia(*red_rgb)
        protanomaly = analyzer._simulate_protanomaly(*red_rgb)
        deuteranomaly = analyzer._simulate_deuteranomaly(*red_rgb)
        tritanomaly = analyzer._simulate_tritanomaly(*red_rgb)
        
        # All should return valid RGB tuples
        for simulation in [protanopia, deuteranopia, tritanopia, protanomaly, deuteranomaly, tritanomaly]:
            assert len(simulation) == 3
            assert all(0 <= c <= 255 for c in simulation)
    
    def test_distinguishability_calculation(self):
        """Test color distinguishability calculation."""
        analyzer = ColorAnalyzer()
        
        # Test with very different colors
        colors_different = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        distinguishability_high = analyzer._calculate_distinguishability(colors_different)
        assert 0.5 <= distinguishability_high <= 1.0
        
        # Test with similar colors
        colors_similar = [(100, 100, 100), (101, 101, 101), (102, 102, 102)]
        distinguishability_low = analyzer._calculate_distinguishability(colors_similar)
        assert distinguishability_low < 0.1
        
        # Test with single color
        single_color = [(255, 0, 0)]
        distinguishability_single = analyzer._calculate_distinguishability(single_color)
        assert distinguishability_single == 1.0
    
    def test_color_blindness_recommendations(self):
        """Test color blindness recommendation generation."""
        analyzer = ColorAnalyzer()
        
        # Test with good distinguishability
        good_scores = {
            "protanopia": 0.8,
            "deuteranopia": 0.9, 
            "tritanopia": 0.85
        }
        recommendations = analyzer._generate_cb_recommendations(good_scores)
        assert any("good accessibility" in rec.lower() for rec in recommendations)
        
        # Test with poor distinguishability
        poor_scores = {
            "protanopia": 0.2,
            "deuteranopia": 0.1,
            "tritanopia": 0.15
        }
        recommendations = analyzer._generate_cb_recommendations(poor_scores)
        assert len(recommendations) > 1  # Should have multiple recommendations
        assert any("pattern" in rec.lower() or "contrast" in rec.lower() for rec in recommendations)
