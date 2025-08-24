"""Advanced color analysis and color theory utilities."""

import math
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from PIL import Image
import colorsys
from colormath.color_objects import sRGBColor, LabColor, XYZColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cie1994, delta_e_cie1976

from ..models.color_data import ColorInfo, ColorHarmony


class ColorAnalyzer:
    """Advanced color analysis with color theory and perception algorithms."""
    
    def __init__(self):
        """Initialize the color analyzer."""
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def analyze_color_relationships(self, colors: List[ColorInfo]) -> Dict[str, any]:
        """Analyze relationships between colors in a palette."""
        if len(colors) < 2:
            return {"error": "Need at least 2 colors for relationship analysis"}
        
        relationships = {
            "harmony_types": self._detect_all_harmonies(colors),
            "color_distances": self._calculate_color_distances(colors),
            "perceptual_groupings": self._group_by_perception(colors),
            "complementary_pairs": self._find_complementary_pairs(colors),
            "analogous_groups": self._find_analogous_groups(colors),
            "temperature_analysis": self._analyze_temperature_relationships(colors),
            "contrast_matrix": self._create_contrast_matrix(colors),
            "accessibility_pairs": self._find_accessible_pairs(colors)
        }
        
        return relationships
    
    def calculate_color_emotion(self, colors: List[ColorInfo]) -> Dict[str, float]:
        """Calculate emotional associations of colors based on color psychology."""
        emotions = {
            "energy": 0.0,
            "calmness": 0.0, 
            "warmth": 0.0,
            "sophistication": 0.0,
            "playfulness": 0.0,
            "trustworthiness": 0.0,
            "creativity": 0.0,
            "luxury": 0.0
        }
        
        total_weight = sum(color.percentage for color in colors)
        
        for color in colors:
            weight = color.percentage / total_weight
            h, s, v = color.hsv
            h_deg = h * 360
            
            # Energy (bright, saturated colors)
            energy_score = s * v * (1.0 if h_deg < 60 or h_deg > 300 else 0.7)
            emotions["energy"] += energy_score * weight
            
            # Calmness (cool, desaturated colors)
            calmness_score = (1 - s) * (1.0 if 180 <= h_deg <= 240 else 0.5)
            emotions["calmness"] += calmness_score * weight
            
            # Warmth (reds, oranges, yellows)
            warmth_score = 1.0 if h_deg < 60 or h_deg > 300 else 0.2
            emotions["warmth"] += warmth_score * weight
            
            # Sophistication (dark, desaturated colors)
            soph_score = (1 - v) * (1 - s) + (0.5 if s > 0.8 and v < 0.3 else 0)
            emotions["sophistication"] += soph_score * weight
            
            # Playfulness (bright, saturated, varied hues)
            play_score = s * v * (1.0 if s > 0.7 and v > 0.7 else 0.3)
            emotions["playfulness"] += play_score * weight
            
            # Trustworthiness (blues, stable colors)
            trust_score = 1.0 if 200 <= h_deg <= 260 else 0.4
            trust_score *= (1 - abs(s - 0.5))  # Moderate saturation
            emotions["trustworthiness"] += trust_score * weight
            
            # Creativity (purple, unusual combinations)
            creative_score = 1.0 if 270 <= h_deg <= 330 else 0.6
            creative_score *= s  # Higher for more saturated
            emotions["creativity"] += creative_score * weight
            
            # Luxury (deep colors, gold, silver)
            luxury_score = 0.0
            if v < 0.3 and s > 0.5:  # Deep saturated colors
                luxury_score = 0.8
            elif 45 <= h_deg <= 75 and s > 0.7 and v > 0.5:  # Gold
                luxury_score = 1.0
            elif s < 0.1 and (v > 0.8 or v < 0.2):  # Silver/Black
                luxury_score = 0.7
            emotions["luxury"] += luxury_score * weight
        
        # Normalize to 0-1 range
        for emotion in emotions:
            emotions[emotion] = max(0.0, min(1.0, emotions[emotion]))
        
        return emotions
    
    def analyze_brand_personality(self, colors: List[ColorInfo]) -> Dict[str, any]:
        """Analyze brand personality traits based on color psychology."""
        if not colors:
            return {"error": "No colors provided"}
        
        personality_traits = {
            "modern": 0.0,
            "traditional": 0.0,
            "professional": 0.0,
            "casual": 0.0,
            "bold": 0.0,
            "subtle": 0.0,
            "masculine": 0.0,
            "feminine": 0.0,
            "youthful": 0.0,
            "mature": 0.0
        }
        
        total_weight = sum(color.percentage for color in colors)
        
        for color in colors:
            weight = color.percentage / total_weight
            h, s, v = color.hsv
            h_deg = h * 360
            
            # Modern (bright, saturated, tech colors)
            modern_score = s * (1.0 if 180 <= h_deg <= 240 or s > 0.8 else 0.5)
            personality_traits["modern"] += modern_score * weight
            
            # Traditional (earth tones, classic colors)
            traditional_score = 1.0 if (0 <= h_deg <= 30 or 300 <= h_deg <= 360) and s < 0.7 else 0.3
            personality_traits["traditional"] += traditional_score * weight
            
            # Professional (blues, grays, conservative)
            prof_score = 1.0 if 200 <= h_deg <= 260 or s < 0.2 else 0.4
            personality_traits["professional"] += prof_score * weight
            
            # Casual (bright, varied)
            casual_score = s * v
            personality_traits["casual"] += casual_score * weight
            
            # Bold (high saturation, contrast)
            bold_score = s * (1.0 if s > 0.8 else 0.5)
            personality_traits["bold"] += bold_score * weight
            
            # Subtle (low saturation, harmonious)
            subtle_score = (1 - s) * (1.0 if s < 0.3 else 0.5)
            personality_traits["subtle"] += subtle_score * weight
            
            # Masculine (darker, cooler colors)
            masc_score = (1 - v) * (1.0 if 200 <= h_deg <= 300 else 0.6)
            personality_traits["masculine"] += masc_score * weight
            
            # Feminine (lighter, warmer, pinks)
            fem_score = v * (1.0 if 300 <= h_deg <= 360 else 0.6)
            personality_traits["feminine"] += fem_score * weight
            
            # Youthful (bright, saturated)
            youth_score = s * v
            personality_traits["youthful"] += youth_score * weight
            
            # Mature (muted, sophisticated)
            mature_score = (1 - s) * (1.0 if 0.3 <= v <= 0.7 else 0.5)
            personality_traits["mature"] += mature_score * weight
        
        # Normalize
        for trait in personality_traits:
            personality_traits[trait] = max(0.0, min(1.0, personality_traits[trait]))
        
        # Determine dominant traits
        dominant_traits = sorted(personality_traits.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "personality_scores": personality_traits,
            "dominant_traits": [{"trait": trait, "score": score} for trait, score in dominant_traits],
            "overall_personality": self._classify_overall_personality(personality_traits)
        }
    
    def generate_color_suggestions(self, colors: List[ColorInfo], suggestion_type: str = "harmony") -> List[ColorInfo]:
        """Generate color suggestions based on existing palette."""
        if not colors:
            return []
        
        base_color = colors[0]  # Use most frequent color as base
        h, s, v = base_color.hsv
        h_deg = h * 360
        
        suggestions = []
        
        if suggestion_type == "harmony":
            # Complementary
            comp_h = (h_deg + 180) % 360
            suggestions.append(self._hsv_to_colorinfo(comp_h/360, s, v))
            
            # Triadic
            tri_h1 = (h_deg + 120) % 360
            tri_h2 = (h_deg + 240) % 360
            suggestions.extend([
                self._hsv_to_colorinfo(tri_h1/360, s, v),
                self._hsv_to_colorinfo(tri_h2/360, s, v)
            ])
            
            # Split complementary
            split_h1 = (h_deg + 150) % 360
            split_h2 = (h_deg + 210) % 360
            suggestions.extend([
                self._hsv_to_colorinfo(split_h1/360, s, v),
                self._hsv_to_colorinfo(split_h2/360, s, v)
            ])
        
        elif suggestion_type == "monochromatic":
            # Same hue, different saturations and values
            for s_offset in [-0.3, -0.1, 0.1, 0.3]:
                for v_offset in [-0.2, 0.2]:
                    new_s = max(0, min(1, s + s_offset))
                    new_v = max(0, min(1, v + v_offset))
                    suggestions.append(self._hsv_to_colorinfo(h, new_s, new_v))
        
        elif suggestion_type == "analogous":
            # Adjacent hues
            for h_offset in [-30, -15, 15, 30, 45]:
                new_h = (h_deg + h_offset) % 360
                suggestions.append(self._hsv_to_colorinfo(new_h/360, s, v))
        
        elif suggestion_type == "accessibility":
            # Generate high contrast colors for accessibility
            for target_color in colors[:3]:
                accessible_color = self._generate_accessible_color(target_color)
                if accessible_color:
                    suggestions.append(accessible_color)
        
        return suggestions[:8]  # Limit to 8 suggestions
    
    def calculate_color_blindness_impact(self, colors: List[ColorInfo]) -> Dict[str, any]:
        """Analyze how colors appear to people with different types of color blindness."""
        simulations = {
            "protanopia": [],    # Red-blind
            "deuteranopia": [],  # Green-blind  
            "tritanopia": [],    # Blue-blind
            "protanomaly": [],   # Red-weak
            "deuteranomaly": [], # Green-weak
            "tritanomaly": []    # Blue-weak
        }
        
        for color in colors:
            r, g, b = [c/255.0 for c in color.rgb]
            
            # Simulate different types of color blindness
            simulations["protanopia"].append(self._simulate_protanopia(r, g, b))
            simulations["deuteranopia"].append(self._simulate_deuteranopia(r, g, b))
            simulations["tritanopia"].append(self._simulate_tritanopia(r, g, b))
            simulations["protanomaly"].append(self._simulate_protanomaly(r, g, b))
            simulations["deuteranomaly"].append(self._simulate_deuteranomaly(r, g, b))
            simulations["tritanomaly"].append(self._simulate_tritanomaly(r, g, b))
        
        # Calculate distinguishability scores
        distinguishability = {}
        for cb_type, sim_colors in simulations.items():
            score = self._calculate_distinguishability(sim_colors)
            distinguishability[cb_type] = score
        
        return {
            "simulations": simulations,
            "distinguishability_scores": distinguishability,
            "overall_accessibility": min(distinguishability.values()),
            "recommendations": self._generate_cb_recommendations(distinguishability)
        }
    
    def _detect_all_harmonies(self, colors: List[ColorInfo]) -> List[str]:
        """Detect all types of color harmonies present."""
        harmonies = []
        
        if len(colors) < 2:
            return harmonies
        
        # Convert to HSV for analysis
        hsv_colors = [(c.hsv[0] * 360, c.hsv[1], c.hsv[2]) for c in colors]
        
        # Check for monochromatic
        if self._is_monochromatic(hsv_colors):
            harmonies.append("monochromatic")
        
        # Check for complementary
        if self._is_complementary(hsv_colors):
            harmonies.append("complementary")
        
        # Check for triadic
        if self._is_triadic(hsv_colors):
            harmonies.append("triadic")
        
        # Check for analogous
        if self._is_analogous(hsv_colors):
            harmonies.append("analogous")
        
        # Check for split complementary
        if self._is_split_complementary(hsv_colors):
            harmonies.append("split_complementary")
        
        # Check for tetradic
        if self._is_tetradic(hsv_colors):
            harmonies.append("tetradic")
        
        return harmonies
    
    def _calculate_color_distances(self, colors: List[ColorInfo]) -> Dict[str, float]:
        """Calculate various color distance metrics."""
        if len(colors) < 2:
            return {}
        
        distances = {
            "euclidean_rgb": [],
            "euclidean_lab": [],
            "delta_e_cie2000": [],
            "delta_e_cie1994": [],
            "delta_e_cie1976": []
        }
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                c1, c2 = colors[i], colors[j]
                
                # RGB Euclidean distance
                rgb_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1.rgb, c2.rgb)))
                distances["euclidean_rgb"].append(rgb_dist)
                
                # LAB distances
                if c1.lab and c2.lab:
                    lab_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c1.lab, c2.lab)))
                    distances["euclidean_lab"].append(lab_dist)
                    
                    # Create Lab color objects for Delta E calculations
                    lab1 = LabColor(*c1.lab)
                    lab2 = LabColor(*c2.lab)
                    
                    distances["delta_e_cie2000"].append(delta_e_cie2000(lab1, lab2))
                    distances["delta_e_cie1994"].append(delta_e_cie1994(lab1, lab2))
                    distances["delta_e_cie1976"].append(delta_e_cie1976(lab1, lab2))
        
        # Calculate statistics
        result = {}
        for metric, values in distances.items():
            if values:
                result[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "std": np.std(values).item() if len(values) > 1 else 0
                }
        
        return result
    
    def _group_by_perception(self, colors: List[ColorInfo]) -> Dict[str, List[ColorInfo]]:
        """Group colors by perceptual similarity."""
        groups = {
            "reds": [],
            "oranges": [],
            "yellows": [],
            "greens": [],
            "blues": [],
            "purples": [],
            "pinks": [],
            "browns": [],
            "grays": [],
            "whites": [],
            "blacks": []
        }
        
        for color in colors:
            h, s, v = color.hsv
            h_deg = h * 360
            
            if s < 0.15:  # Low saturation
                if v > 0.9:
                    groups["whites"].append(color)
                elif v < 0.15:
                    groups["blacks"].append(color)
                else:
                    groups["grays"].append(color)
            else:  # Colored
                if 0 <= h_deg < 15 or 345 <= h_deg <= 360:
                    groups["reds"].append(color)
                elif 15 <= h_deg < 45:
                    groups["oranges"].append(color)
                elif 45 <= h_deg < 75:
                    groups["yellows"].append(color)
                elif 75 <= h_deg < 150:
                    groups["greens"].append(color)
                elif 150 <= h_deg < 225:
                    groups["blues"].append(color)
                elif 225 <= h_deg < 285:
                    groups["purples"].append(color)
                elif 285 <= h_deg < 345:
                    groups["pinks"].append(color)
                
                # Check for browns (low saturation oranges/yellows)
                if 20 <= h_deg < 60 and s < 0.6 and v < 0.6:
                    groups["browns"].append(color)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _find_complementary_pairs(self, colors: List[ColorInfo]) -> List[Tuple[ColorInfo, ColorInfo]]:
        """Find complementary color pairs."""
        pairs = []
        
        for i, color1 in enumerate(colors):
            for color2 in colors[i+1:]:
                h1 = color1.hsv[0] * 360
                h2 = color2.hsv[0] * 360
                
                # Check if colors are approximately complementary (180° apart)
                hue_diff = abs(h1 - h2)
                hue_diff = min(hue_diff, 360 - hue_diff)
                
                if 160 <= hue_diff <= 200:  # Allow some tolerance
                    pairs.append((color1, color2))
        
        return pairs
    
    def _find_analogous_groups(self, colors: List[ColorInfo]) -> List[List[ColorInfo]]:
        """Find groups of analogous colors."""
        groups = []
        used_colors = set()
        
        for color in colors:
            if id(color) in used_colors:
                continue
                
            h1 = color.hsv[0] * 360
            group = [color]
            used_colors.add(id(color))
            
            # Find other colors within 60° hue range
            for other_color in colors:
                if id(other_color) in used_colors:
                    continue
                    
                h2 = other_color.hsv[0] * 360
                hue_diff = abs(h1 - h2)
                hue_diff = min(hue_diff, 360 - hue_diff)
                
                if hue_diff <= 60:
                    group.append(other_color)
                    used_colors.add(id(other_color))
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _analyze_temperature_relationships(self, colors: List[ColorInfo]) -> Dict[str, any]:
        """Analyze temperature relationships between colors."""
        warm_colors = []
        cool_colors = []
        neutral_colors = []
        
        for color in colors:
            h_deg = color.hsv[0] * 360
            
            if 0 <= h_deg < 60 or 300 <= h_deg <= 360:
                warm_colors.append(color)
            elif 120 <= h_deg < 240:
                cool_colors.append(color)
            else:
                neutral_colors.append(color)
        
        warm_weight = sum(c.percentage for c in warm_colors)
        cool_weight = sum(c.percentage for c in cool_colors)
        neutral_weight = sum(c.percentage for c in neutral_colors)
        
        return {
            "warm_colors": warm_colors,
            "cool_colors": cool_colors,
            "neutral_colors": neutral_colors,
            "warm_percentage": warm_weight,
            "cool_percentage": cool_weight,
            "neutral_percentage": neutral_weight,
            "temperature_balance": "warm" if warm_weight > cool_weight else "cool" if cool_weight > warm_weight else "balanced"
        }
    
    def _create_contrast_matrix(self, colors: List[ColorInfo]) -> List[List[float]]:
        """Create a matrix of contrast ratios between all color pairs."""
        n = len(colors)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    l1 = colors[i].luminance or self._calculate_luminance(*colors[i].rgb)
                    l2 = colors[j].luminance or self._calculate_luminance(*colors[j].rgb)
                    
                    lighter = max(l1, l2)
                    darker = min(l1, l2)
                    
                    matrix[i][j] = (lighter + 0.05) / (darker + 0.05)
                else:
                    matrix[i][j] = 1.0
        
        return matrix
    
    def _find_accessible_pairs(self, colors: List[ColorInfo]) -> List[Tuple[ColorInfo, ColorInfo, float]]:
        """Find color pairs that meet accessibility guidelines."""
        accessible_pairs = []
        
        for i, color1 in enumerate(colors):
            for color2 in colors[i+1:]:
                l1 = color1.luminance or self._calculate_luminance(*color1.rgb)
                l2 = color2.luminance or self._calculate_luminance(*color2.rgb)
                
                lighter = max(l1, l2)
                darker = min(l1, l2)
                
                contrast_ratio = (lighter + 0.05) / (darker + 0.05)
                
                if contrast_ratio >= 4.5:  # WCAG AA standard
                    accessible_pairs.append((color1, color2, contrast_ratio))
        
        return sorted(accessible_pairs, key=lambda x: x[2], reverse=True)
    
    def _classify_overall_personality(self, traits: Dict[str, float]) -> str:
        """Classify overall brand personality based on trait scores."""
        max_trait = max(traits, key=traits.get)
        max_score = traits[max_trait]
        
        if max_score < 0.3:
            return "neutral"
        elif max_score > 0.7:
            return f"strongly_{max_trait}"
        else:
            return max_trait
    
    def _hsv_to_colorinfo(self, h: float, s: float, v: float) -> ColorInfo:
        """Convert HSV to ColorInfo object."""
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        
        return ColorInfo(
            rgb=rgb,
            hex=hex_color,
            hsv=(h, s, v),
            frequency=0,
            percentage=0.0
        )
    
    def _generate_accessible_color(self, target_color: ColorInfo) -> Optional[ColorInfo]:
        """Generate an accessible color that contrasts well with the target."""
        target_luminance = target_color.luminance or self._calculate_luminance(*target_color.rgb)
        
        # If target is dark, generate light color; if light, generate dark
        if target_luminance < 0.18:
            # Generate light color
            new_rgb = (240, 240, 240)
        else:
            # Generate dark color
            new_rgb = (30, 30, 30)
        
        return ColorInfo(
            rgb=new_rgb,
            hex="#{:02x}{:02x}{:02x}".format(*new_rgb),
            frequency=0,
            percentage=0.0
        )
    
    def _calculate_luminance(self, r: int, g: int, b: int) -> float:
        """Calculate relative luminance."""
        def gamma_correct(value):
            value = value / 255.0
            if value <= 0.03928:
                return value / 12.92
            return pow((value + 0.055) / 1.055, 2.4)
        
        r_lin = gamma_correct(r)
        g_lin = gamma_correct(g)
        b_lin = gamma_correct(b)
        
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    
    def _is_monochromatic(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form a monochromatic harmony."""
        if len(hsv_colors) < 2:
            return False
        
        base_hue = hsv_colors[0][0]
        tolerance = 15  # degrees
        
        for h, s, v in hsv_colors[1:]:
            hue_diff = abs(h - base_hue)
            hue_diff = min(hue_diff, 360 - hue_diff)
            if hue_diff > tolerance:
                return False
        
        return True
    
    def _is_complementary(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form a complementary harmony."""
        for i, (h1, s1, v1) in enumerate(hsv_colors):
            for h2, s2, v2 in hsv_colors[i+1:]:
                hue_diff = abs(h1 - h2)
                hue_diff = min(hue_diff, 360 - hue_diff)
                if 160 <= hue_diff <= 200:
                    return True
        return False
    
    def _is_triadic(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form a triadic harmony."""
        if len(hsv_colors) < 3:
            return False
        
        hues = [h for h, s, v in hsv_colors]
        hues.sort()
        
        # Check for roughly 120-degree separations
        for i in range(len(hues) - 2):
            diff1 = hues[i+1] - hues[i]
            diff2 = hues[i+2] - hues[i+1]
            
            if 100 <= diff1 <= 140 and 100 <= diff2 <= 140:
                return True
        
        return False
    
    def _is_analogous(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form an analogous harmony."""
        if len(hsv_colors) < 2:
            return False
        
        hues = [h for h, s, v in hsv_colors]
        hues.sort()
        
        # Check if all hues are within 60 degrees of each other
        return (hues[-1] - hues[0]) <= 60
    
    def _is_split_complementary(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form a split complementary harmony."""
        for h1, s1, v1 in hsv_colors:
            comp_hue = (h1 + 180) % 360
            split1 = (comp_hue - 30) % 360
            split2 = (comp_hue + 30) % 360
            
            found_split1 = any(abs(h - split1) <= 15 or abs(h - split1) >= 345 for h, s, v in hsv_colors)
            found_split2 = any(abs(h - split2) <= 15 or abs(h - split2) >= 345 for h, s, v in hsv_colors)
            
            if found_split1 and found_split2:
                return True
        
        return False
    
    def _is_tetradic(self, hsv_colors: List[Tuple[float, float, float]]) -> bool:
        """Check if colors form a tetradic (square) harmony."""
        if len(hsv_colors) < 4:
            return False
        
        hues = [h for h, s, v in hsv_colors]
        hues.sort()
        
        # Check for roughly 90-degree separations
        for i in range(len(hues) - 3):
            diffs = [hues[i+j+1] - hues[i+j] for j in range(3)]
            if all(75 <= diff <= 105 for diff in diffs):
                return True
        
        return False
    
    # Color blindness simulation methods
    def _simulate_protanopia(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate protanopia (red-blind)."""
        # Simplified simulation matrix
        new_r = 0.567 * r + 0.433 * g
        new_g = 0.558 * r + 0.442 * g  
        new_b = 0.242 * g + 0.758 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _simulate_deuteranopia(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate deuteranopia (green-blind)."""
        new_r = 0.625 * r + 0.375 * g
        new_g = 0.7 * r + 0.3 * g
        new_b = 0.3 * g + 0.7 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _simulate_tritanopia(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate tritanopia (blue-blind)."""
        new_r = 0.95 * r + 0.05 * g
        new_g = 0.433 * g + 0.567 * b
        new_b = 0.475 * g + 0.525 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _simulate_protanomaly(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate protanomaly (red-weak)."""
        new_r = 0.817 * r + 0.183 * g
        new_g = 0.333 * r + 0.667 * g
        new_b = 0.125 * g + 0.875 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _simulate_deuteranomaly(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate deuteranomaly (green-weak)."""
        new_r = 0.8 * r + 0.2 * g
        new_g = 0.258 * r + 0.742 * g
        new_b = 0.142 * g + 0.858 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _simulate_tritanomaly(self, r: float, g: float, b: float) -> Tuple[int, int, int]:
        """Simulate tritanomaly (blue-weak)."""
        new_r = 0.967 * r + 0.033 * g
        new_g = 0.733 * g + 0.267 * b
        new_b = 0.183 * g + 0.817 * b
        return (int(new_r * 255), int(new_g * 255), int(new_b * 255))
    
    def _calculate_distinguishability(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate how distinguishable colors are from each other."""
        if len(colors) < 2:
            return 1.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(colors[i], colors[j])))
                total_distance += distance
                comparisons += 1
        
        avg_distance = total_distance / comparisons
        # Normalize to 0-1 scale (max RGB distance is ~441)
        return min(1.0, avg_distance / 441.0)
    
    def _generate_cb_recommendations(self, distinguishability: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving color blindness accessibility."""
        recommendations = []
        
        min_score = min(distinguishability.values())
        
        if min_score < 0.3:
            recommendations.append("Consider using patterns or textures in addition to colors")
            recommendations.append("Increase contrast between similar colors")
        
        if distinguishability.get("protanopia", 1.0) < 0.4:
            recommendations.append("Add more distinction for red-green color blindness")
        
        if distinguishability.get("tritanopia", 1.0) < 0.4:
            recommendations.append("Improve blue-yellow color distinctions")
        
        if not recommendations:
            recommendations.append("Color palette has good accessibility for color blindness")
        
        return recommendations
