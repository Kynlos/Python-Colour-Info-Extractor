"""Advanced palette generation with multiple styles and algorithms."""

import math
import random
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

from ..models.color_data import ColorInfo, PaletteData


class PaletteGenerator:
    """Advanced palette generation with various styles and techniques."""
    
    def __init__(self):
        """Initialize the palette generator."""
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def generate_palette_variations(
        self, 
        base_colors: List[ColorInfo], 
        style: str = "dominant",
        count: int = 5
    ) -> List[PaletteData]:
        """
        Generate multiple palette variations from base colors.
        
        Args:
            base_colors: Base colors to work with
            style: Style of palette generation
            count: Number of colors in generated palettes
            
        Returns:
            List of generated palettes
        """
        palettes = []
        timestamp = "2024-01-01 00:00:00"  # Placeholder
        
        if style == "dominant":
            palettes.extend(self._generate_dominant_palettes(base_colors, count, timestamp))
        elif style == "harmony":
            palettes.extend(self._generate_harmony_palettes(base_colors, count, timestamp))
        elif style == "monochromatic":
            palettes.extend(self._generate_monochromatic_palettes(base_colors, count, timestamp))
        elif style == "analogous":
            palettes.extend(self._generate_analogous_palettes(base_colors, count, timestamp))
        elif style == "complementary":
            palettes.extend(self._generate_complementary_palettes(base_colors, count, timestamp))
        elif style == "split_complementary":
            palettes.extend(self._generate_split_complementary_palettes(base_colors, count, timestamp))
        elif style == "triadic":
            palettes.extend(self._generate_triadic_palettes(base_colors, count, timestamp))
        elif style == "tetradic":
            palettes.extend(self._generate_tetradic_palettes(base_colors, count, timestamp))
        elif style == "vibrant":
            palettes.extend(self._generate_vibrant_palettes(base_colors, count, timestamp))
        elif style == "muted":
            palettes.extend(self._generate_muted_palettes(base_colors, count, timestamp))
        elif style == "pastel":
            palettes.extend(self._generate_pastel_palettes(base_colors, count, timestamp))
        elif style == "dark":
            palettes.extend(self._generate_dark_palettes(base_colors, count, timestamp))
        elif style == "light":
            palettes.extend(self._generate_light_palettes(base_colors, count, timestamp))
        elif style == "gradient":
            palettes.extend(self._generate_gradient_palettes(base_colors, count, timestamp))
        elif style == "material":
            palettes.extend(self._generate_material_palettes(base_colors, count, timestamp))
        elif style == "web_safe":
            palettes.extend(self._generate_web_safe_palettes(base_colors, count, timestamp))
        else:
            # Default: generate multiple styles
            palettes.extend(self._generate_dominant_palettes(base_colors, count, timestamp))
            palettes.extend(self._generate_harmony_palettes(base_colors, count, timestamp))
        
        return palettes
    
    def create_palette_image(
        self, 
        colors: List[ColorInfo], 
        style: str = "swatches",
        width: int = 800,
        height: int = 200,
        show_labels: bool = True
    ) -> Image.Image:
        """
        Create a visual palette image.
        
        Args:
            colors: Colors to display
            style: Visual style (swatches, gradient, circle, etc.)
            width: Image width
            height: Image height
            show_labels: Whether to show color labels
            
        Returns:
            PIL Image of the palette
        """
        if style == "swatches":
            return self._create_swatch_image(colors, width, height, show_labels)
        elif style == "gradient":
            return self._create_gradient_image(colors, width, height)
        elif style == "circle":
            return self._create_circle_image(colors, width, height, show_labels)
        elif style == "hexagon":
            return self._create_hexagon_image(colors, width, height)
        elif style == "squares":
            return self._create_squares_image(colors, width, height, show_labels)
        elif style == "waves":
            return self._create_waves_image(colors, width, height)
        else:
            return self._create_swatch_image(colors, width, height, show_labels)
    
    def analyze_palette_mood(self, colors: List[ColorInfo]) -> Dict[str, Any]:
        """
        Analyze the mood and characteristics of a color palette.
        
        Args:
            colors: Colors in the palette
            
        Returns:
            Dictionary with mood analysis
        """
        if not colors:
            return {"error": "No colors provided"}
        
        # Calculate overall characteristics
        total_weight = sum(c.percentage for c in colors)
        
        avg_hue = 0
        avg_saturation = 0
        avg_value = 0
        
        for color in colors:
            weight = color.percentage / total_weight if total_weight > 0 else 1 / len(colors)
            h, s, v = color.hsv or (0, 0, 0)
            
            avg_hue += h * weight
            avg_saturation += s * weight
            avg_value += v * weight
        
        # Determine mood characteristics
        mood_analysis = {
            "overall_mood": self._determine_overall_mood(colors),
            "temperature": self._analyze_temperature(colors),
            "energy_level": avg_saturation * avg_value,
            "sophistication": self._calculate_sophistication(colors),
            "harmony_score": self._calculate_harmony_score(colors),
            "contrast_level": self._calculate_contrast_level(colors),
            "color_balance": self._analyze_color_balance(colors),
            "seasonal_association": self._determine_seasonal_association(colors),
            "style_recommendations": self._get_style_recommendations(colors)
        }
        
        return mood_analysis
    
    def suggest_palette_improvements(self, colors: List[ColorInfo]) -> List[Dict[str, Any]]:
        """
        Suggest improvements to an existing palette.
        
        Args:
            colors: Current palette colors
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze current palette
        if not colors:
            return [{"type": "error", "message": "No colors provided"}]
        
        # Check contrast
        contrast_issues = self._check_contrast_issues(colors)
        if contrast_issues:
            suggestions.append({
                "type": "contrast",
                "priority": "high",
                "message": "Some colors have poor contrast ratios",
                "details": contrast_issues,
                "solution": "Adjust lightness values or add darker/lighter variants"
            })
        
        # Check harmony
        harmony_score = self._calculate_harmony_score(colors)
        if harmony_score < 0.6:
            suggestions.append({
                "type": "harmony",
                "priority": "medium",
                "message": "Palette could have better color harmony",
                "score": harmony_score,
                "solution": "Consider using colors based on color theory relationships"
            })
        
        # Check balance
        balance_analysis = self._analyze_color_balance(colors)
        if balance_analysis.get("imbalance_score", 0) > 0.7:
            suggestions.append({
                "type": "balance",
                "priority": "medium",
                "message": "Color palette may be unbalanced",
                "details": balance_analysis,
                "solution": "Add colors from underrepresented areas of the color wheel"
            })
        
        # Check accessibility
        accessibility_issues = self._check_accessibility_issues(colors)
        if accessibility_issues:
            suggestions.append({
                "type": "accessibility",
                "priority": "high",
                "message": "Palette may have accessibility issues",
                "details": accessibility_issues,
                "solution": "Ensure sufficient contrast for text and UI elements"
            })
        
        # Suggest additional colors
        missing_colors = self._suggest_missing_colors(colors)
        if missing_colors:
            suggestions.append({
                "type": "enhancement",
                "priority": "low",
                "message": "Consider adding these colors to enhance the palette",
                "suggested_colors": missing_colors,
                "solution": "Add accent or neutral colors to increase versatility"
            })
        
        return suggestions
    
    # Private methods for palette generation
    
    def _generate_dominant_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate palettes based on dominant colors."""
        palettes = []
        
        # Main dominant palette
        dominant_colors = base_colors[:count]
        palettes.append(PaletteData(
            name="Dominant Colors",
            colors=dominant_colors,
            palette_type="dominant",
            created_at=timestamp
        ))
        
        # High frequency palette
        high_freq_colors = [c for c in base_colors if c.percentage > 5.0][:count]
        if len(high_freq_colors) >= 3:
            palettes.append(PaletteData(
                name="High Frequency Colors", 
                colors=high_freq_colors,
                palette_type="frequency",
                created_at=timestamp
            ))
        
        return palettes
    
    def _generate_harmony_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate harmonious color palettes."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        h_deg = h * 360
        
        # Complementary palette
        comp_colors = [base_color]
        comp_h = (h_deg + 180) % 360
        comp_colors.append(self._create_color_info_hsv(comp_h / 360, s, v))
        
        # Add variations
        for offset in [30, 60]:
            if len(comp_colors) < count:
                var_h = (comp_h + offset) % 360
                comp_colors.append(self._create_color_info_hsv(var_h / 360, s * 0.8, v * 1.1))
        
        palettes.append(PaletteData(
            name="Complementary Harmony",
            colors=comp_colors[:count],
            palette_type="complementary",
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_monochromatic_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate monochromatic palette variations."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        
        # Light monochromatic
        light_colors = []
        for i in range(count):
            new_v = min(1.0, v + (i * 0.15))
            new_s = max(0.1, s - (i * 0.1))
            light_colors.append(self._create_color_info_hsv(h, new_s, new_v))
        
        palettes.append(PaletteData(
            name="Light Monochromatic",
            colors=light_colors,
            palette_type="monochromatic_light",
            created_at=timestamp
        ))
        
        # Dark monochromatic
        dark_colors = []
        for i in range(count):
            new_v = max(0.1, v - (i * 0.15))
            new_s = min(1.0, s + (i * 0.1))
            dark_colors.append(self._create_color_info_hsv(h, new_s, new_v))
        
        palettes.append(PaletteData(
            name="Dark Monochromatic",
            colors=dark_colors,
            palette_type="monochromatic_dark", 
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_analogous_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate analogous color palettes."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        h_deg = h * 360
        
        analog_colors = [base_color]
        
        # Add colors at 30-degree intervals
        for i in range(1, count):
            offset = (i % 2) * 60 - 30  # Alternate between -30 and +30 degrees
            if i > 2:
                offset = ((i - 2) % 2) * 90 - 45  # Then -45 and +45 degrees
            
            new_h = (h_deg + offset) % 360
            analog_colors.append(self._create_color_info_hsv(new_h / 360, s, v))
        
        palettes.append(PaletteData(
            name="Analogous Colors",
            colors=analog_colors,
            palette_type="analogous",
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_complementary_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate complementary color palettes."""
        return self._generate_harmony_palettes(base_colors, count, timestamp)
    
    def _generate_split_complementary_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate split complementary palettes."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        h_deg = h * 360
        
        split_colors = [base_color]
        
        # Split complementary colors (150° and 210° from base)
        comp_h = (h_deg + 180) % 360
        split1_h = (comp_h - 30) % 360
        split2_h = (comp_h + 30) % 360
        
        split_colors.append(self._create_color_info_hsv(split1_h / 360, s, v))
        split_colors.append(self._create_color_info_hsv(split2_h / 360, s, v))
        
        # Add variations to fill the count
        while len(split_colors) < count:
            # Add lighter/darker versions
            base_idx = len(split_colors) % 3
            base_hsv = split_colors[base_idx].hsv
            new_v = base_hsv[2] * (0.7 if len(split_colors) % 2 else 1.3)
            new_v = max(0.1, min(1.0, new_v))
            
            split_colors.append(self._create_color_info_hsv(base_hsv[0], base_hsv[1], new_v))
        
        palettes.append(PaletteData(
            name="Split Complementary",
            colors=split_colors[:count],
            palette_type="split_complementary",
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_triadic_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate triadic color palettes."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        h_deg = h * 360
        
        triadic_colors = [base_color]
        
        # Triadic colors (120° apart)
        tri1_h = (h_deg + 120) % 360
        tri2_h = (h_deg + 240) % 360
        
        triadic_colors.append(self._create_color_info_hsv(tri1_h / 360, s, v))
        triadic_colors.append(self._create_color_info_hsv(tri2_h / 360, s, v))
        
        # Add variations
        while len(triadic_colors) < count:
            base_idx = len(triadic_colors) % 3
            base_hsv = triadic_colors[base_idx].hsv
            
            # Vary saturation and value
            new_s = base_hsv[1] * random.uniform(0.6, 1.4)
            new_v = base_hsv[2] * random.uniform(0.7, 1.3)
            new_s = max(0.1, min(1.0, new_s))
            new_v = max(0.1, min(1.0, new_v))
            
            triadic_colors.append(self._create_color_info_hsv(base_hsv[0], new_s, new_v))
        
        palettes.append(PaletteData(
            name="Triadic Colors",
            colors=triadic_colors[:count],
            palette_type="triadic",
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_tetradic_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate tetradic (square) color palettes."""
        palettes = []
        
        if not base_colors:
            return palettes
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        h_deg = h * 360
        
        tetradic_colors = [base_color]
        
        # Tetradic colors (90° apart)
        for offset in [90, 180, 270]:
            if len(tetradic_colors) < count:
                new_h = (h_deg + offset) % 360
                tetradic_colors.append(self._create_color_info_hsv(new_h / 360, s, v))
        
        palettes.append(PaletteData(
            name="Tetradic Square",
            colors=tetradic_colors[:count],
            palette_type="tetradic",
            created_at=timestamp
        ))
        
        return palettes
    
    def _generate_vibrant_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate vibrant color variations."""
        vibrant_colors = []
        
        for color in base_colors:
            if color.hsv and color.hsv[1] > 0.3:  # Already somewhat saturated
                # Increase saturation and adjust value
                h, s, v = color.hsv
                new_s = min(1.0, s * 1.3)
                new_v = max(0.4, min(0.9, v * 0.9))  # Slightly darker to maintain vibrance
                vibrant_colors.append(self._create_color_info_hsv(h, new_s, new_v))
            
            if len(vibrant_colors) >= count:
                break
        
        if vibrant_colors:
            return [PaletteData(
                name="Vibrant Colors",
                colors=vibrant_colors,
                palette_type="vibrant",
                created_at=timestamp
            )]
        
        return []
    
    def _generate_muted_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate muted color variations."""
        muted_colors = []
        
        for color in base_colors:
            if color.hsv:
                h, s, v = color.hsv
                # Reduce saturation for muted effect
                new_s = s * 0.4
                new_v = min(1.0, v * 1.1)  # Slightly lighter
                muted_colors.append(self._create_color_info_hsv(h, new_s, new_v))
            
            if len(muted_colors) >= count:
                break
        
        if muted_colors:
            return [PaletteData(
                name="Muted Colors",
                colors=muted_colors,
                palette_type="muted",
                created_at=timestamp
            )]
        
        return []
    
    def _generate_pastel_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate pastel color variations."""
        pastel_colors = []
        
        for color in base_colors:
            if color.hsv:
                h, s, v = color.hsv
                # Low saturation, high value for pastels
                new_s = min(0.3, s * 0.5)
                new_v = max(0.8, v * 1.2)
                new_v = min(1.0, new_v)
                pastel_colors.append(self._create_color_info_hsv(h, new_s, new_v))
            
            if len(pastel_colors) >= count:
                break
        
        if pastel_colors:
            return [PaletteData(
                name="Pastel Colors",
                colors=pastel_colors,
                palette_type="pastel",
                created_at=timestamp
            )]
        
        return []
    
    def _generate_dark_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate dark color variations."""
        dark_colors = []
        
        for color in base_colors:
            if color.hsv:
                h, s, v = color.hsv
                # Low value for dark colors, maintain or increase saturation
                new_s = min(1.0, s * 1.2)
                new_v = v * 0.4
                dark_colors.append(self._create_color_info_hsv(h, new_s, new_v))
            
            if len(dark_colors) >= count:
                break
        
        if dark_colors:
            return [PaletteData(
                name="Dark Colors",
                colors=dark_colors,
                palette_type="dark",
                created_at=timestamp
            )]
        
        return []
    
    def _generate_light_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate light color variations."""
        light_colors = []
        
        for color in base_colors:
            if color.hsv:
                h, s, v = color.hsv
                # High value, reduced saturation for light colors
                new_s = s * 0.6
                new_v = max(0.8, v * 1.3)
                new_v = min(1.0, new_v)
                light_colors.append(self._create_color_info_hsv(h, new_s, new_v))
            
            if len(light_colors) >= count:
                break
        
        if light_colors:
            return [PaletteData(
                name="Light Colors",
                colors=light_colors,
                palette_type="light",
                created_at=timestamp
            )]
        
        return []
    
    def _generate_gradient_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate gradient-based palettes."""
        if len(base_colors) < 2:
            return []
        
        # Create gradient between two dominant colors
        color1 = base_colors[0]
        color2 = base_colors[1]
        
        gradient_colors = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0
            
            # Linear interpolation in HSV space
            h1, s1, v1 = color1.hsv or (0, 0, 0)
            h2, s2, v2 = color2.hsv or (0, 0, 0)
            
            # Handle hue wrap-around
            h_diff = h2 - h1
            if h_diff > 0.5:
                h_diff -= 1.0
            elif h_diff < -0.5:
                h_diff += 1.0
            
            new_h = (h1 + t * h_diff) % 1.0
            new_s = s1 + t * (s2 - s1)
            new_v = v1 + t * (v2 - v1)
            
            gradient_colors.append(self._create_color_info_hsv(new_h, new_s, new_v))
        
        return [PaletteData(
            name="Gradient Palette",
            colors=gradient_colors,
            palette_type="gradient",
            created_at=timestamp
        )]
    
    def _generate_material_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate Material Design-inspired palettes."""
        if not base_colors:
            return []
        
        base_color = base_colors[0]
        h, s, v = base_color.hsv or (0, 0.5, 0.5)
        
        # Material Design color weights
        material_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        material_colors = []
        
        for i, weight in enumerate(material_weights[:count]):
            # Adjust value based on material design principles
            if weight < 0.5:
                # Lighter shades
                new_v = min(1.0, v + (0.5 - weight) * 0.8)
                new_s = s * (0.5 + weight)
            else:
                # Darker shades
                new_v = v * (1.5 - weight)
                new_s = min(1.0, s * (0.5 + weight * 0.5))
            
            material_colors.append(self._create_color_info_hsv(h, new_s, new_v))
        
        return [PaletteData(
            name="Material Design",
            colors=material_colors,
            palette_type="material",
            created_at=timestamp
        )]
    
    def _generate_web_safe_palettes(self, base_colors: List[ColorInfo], count: int, timestamp: str) -> List[PaletteData]:
        """Generate web-safe color palettes."""
        web_safe_values = [0x00, 0x33, 0x66, 0x99, 0xCC, 0xFF]
        web_safe_colors = []
        
        for color in base_colors:
            # Find closest web-safe color
            r, g, b = color.rgb
            
            safe_r = min(web_safe_values, key=lambda x: abs(x - r))
            safe_g = min(web_safe_values, key=lambda x: abs(x - g))
            safe_b = min(web_safe_values, key=lambda x: abs(x - b))
            
            safe_color = ColorInfo(
                rgb=(safe_r, safe_g, safe_b),
                hex=f"#{safe_r:02x}{safe_g:02x}{safe_b:02x}",
                frequency=color.frequency,
                percentage=color.percentage
            )
            
            web_safe_colors.append(safe_color)
            
            if len(web_safe_colors) >= count:
                break
        
        if web_safe_colors:
            return [PaletteData(
                name="Web Safe Colors",
                colors=web_safe_colors,
                palette_type="web_safe",
                created_at=timestamp
            )]
        
        return []
    
    # Image creation methods
    
    def _create_swatch_image(self, colors: List[ColorInfo], width: int, height: int, show_labels: bool) -> Image.Image:
        """Create a swatch-style palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        swatch_width = width // len(colors)
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        for i, color in enumerate(colors):
            x1 = i * swatch_width
            x2 = x1 + swatch_width
            
            draw.rectangle([(x1, 0), (x2, height)], fill=color.rgb)
            
            if show_labels and height > 40:
                # Add text labels
                text = color.hex
                try:
                    # Try to load a font (this might fail on some systems)
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                text_x = x1 + (swatch_width - text_width) // 2
                text_y = height - text_height - 5
                
                # Choose text color based on background brightness
                brightness = sum(color.rgb) / 3
                text_color = 'black' if brightness > 127 else 'white'
                
                draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        return image
    
    def _create_gradient_image(self, colors: List[ColorInfo], width: int, height: int) -> Image.Image:
        """Create a gradient-style palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        if len(colors) == 1:
            return Image.new('RGB', (width, height), colors[0].rgb)
        
        image = Image.new('RGB', (width, height))
        
        for x in range(width):
            t = x / (width - 1)
            
            # Find the two colors to interpolate between
            segment = t * (len(colors) - 1)
            i = int(segment)
            local_t = segment - i
            
            if i >= len(colors) - 1:
                color = colors[-1].rgb
            else:
                color1 = colors[i].rgb
                color2 = colors[i + 1].rgb
                
                # Linear interpolation
                color = tuple(int(c1 + local_t * (c2 - c1)) for c1, c2 in zip(color1, color2))
            
            for y in range(height):
                image.putpixel((x, y), color)
        
        return image
    
    def _create_circle_image(self, colors: List[ColorInfo], width: int, height: int, show_labels: bool) -> Image.Image:
        """Create a circular palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        angle_step = 360 / len(colors)
        
        for i, color in enumerate(colors):
            start_angle = i * angle_step
            end_angle = (i + 1) * angle_step
            
            # Draw pie slice
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.pieslice(bbox, start_angle, end_angle, fill=color.rgb, outline='white', width=2)
        
        return image
    
    def _create_hexagon_image(self, colors: List[ColorInfo], width: int, height: int) -> Image.Image:
        """Create a hexagonal palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        center_x, center_y = width // 2, height // 2
        hex_size = min(width, height) // 6
        
        # Draw hexagons in a spiral pattern
        for i, color in enumerate(colors[:7]):  # Limit to 7 colors (center + 6 around)
            if i == 0:
                # Center hexagon
                x, y = center_x, center_y
            else:
                # Surrounding hexagons
                angle = (i - 1) * 60  # 60 degrees apart
                radius = hex_size * 1.8
                x = center_x + int(radius * math.cos(math.radians(angle)))
                y = center_y + int(radius * math.sin(math.radians(angle)))
            
            # Draw hexagon
            points = []
            for j in range(6):
                angle = j * 60
                px = x + int(hex_size * math.cos(math.radians(angle)))
                py = y + int(hex_size * math.sin(math.radians(angle)))
                points.append((px, py))
            
            draw.polygon(points, fill=color.rgb, outline='white', width=2)
        
        return image
    
    def _create_squares_image(self, colors: List[ColorInfo], width: int, height: int, show_labels: bool) -> Image.Image:
        """Create a square grid palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        # Calculate grid dimensions
        cols = int(math.ceil(math.sqrt(len(colors))))
        rows = int(math.ceil(len(colors) / cols))
        
        square_width = width // cols
        square_height = height // rows
        
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        for i, color in enumerate(colors):
            row = i // cols
            col = i % cols
            
            x1 = col * square_width
            y1 = row * square_height
            x2 = x1 + square_width
            y2 = y1 + square_height
            
            draw.rectangle([(x1, y1), (x2, y2)], fill=color.rgb, outline='white', width=1)
        
        return image
    
    def _create_waves_image(self, colors: List[ColorInfo], width: int, height: int) -> Image.Image:
        """Create a wave-style palette image."""
        if not colors:
            return Image.new('RGB', (width, height), 'white')
        
        image = Image.new('RGB', (width, height), 'white')
        
        wave_height = height // len(colors)
        
        for i, color in enumerate(colors):
            y_start = i * wave_height
            y_end = y_start + wave_height
            
            for y in range(y_start, min(y_end, height)):
                for x in range(width):
                    # Create wave effect
                    wave_offset = int(20 * math.sin(x * 0.02 + i))
                    actual_y = y + wave_offset
                    
                    if 0 <= actual_y < height:
                        image.putpixel((x, actual_y), color.rgb)
        
        return image
    
    # Helper methods
    
    def _create_color_info_hsv(self, h: float, s: float, v: float) -> ColorInfo:
        """Create ColorInfo from HSV values."""
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        return ColorInfo(
            rgb=rgb,
            hex=hex_color,
            hsv=(h, s, v),
            frequency=0,
            percentage=0.0
        )
    
    def _determine_overall_mood(self, colors: List[ColorInfo]) -> str:
        """Determine the overall mood of the palette."""
        if not colors:
            return "neutral"
        
        # Analyze color characteristics
        warm_count = 0
        cool_count = 0
        dark_count = 0
        bright_count = 0
        saturated_count = 0
        
        for color in colors:
            if color.hsv:
                h, s, v = color.hsv
                h_deg = h * 360
                
                # Temperature
                if 0 <= h_deg < 60 or 300 <= h_deg <= 360:
                    warm_count += 1
                elif 120 <= h_deg < 240:
                    cool_count += 1
                
                # Brightness
                if v > 0.7:
                    bright_count += 1
                elif v < 0.3:
                    dark_count += 1
                
                # Saturation
                if s > 0.6:
                    saturated_count += 1
        
        total_colors = len(colors)
        
        # Determine dominant characteristics
        is_warm = warm_count > cool_count
        is_dark = dark_count / total_colors > 0.4
        is_bright = bright_count / total_colors > 0.4
        is_saturated = saturated_count / total_colors > 0.5
        
        # Determine mood
        if is_dark and not is_saturated:
            return "sophisticated" if not is_warm else "cozy"
        elif is_bright and is_saturated:
            return "energetic" if is_warm else "fresh"
        elif is_bright and not is_saturated:
            return "calm" if not is_warm else "cheerful"
        elif is_saturated:
            return "vibrant" if is_warm else "bold"
        else:
            return "neutral"
    
    def _analyze_temperature(self, colors: List[ColorInfo]) -> Dict[str, Any]:
        """Analyze color temperature characteristics."""
        if not colors:
            return {"overall": "neutral", "warm_ratio": 0, "cool_ratio": 0}
        
        warm_weight = 0
        cool_weight = 0
        neutral_weight = 0
        
        for color in colors:
            if color.hsv:
                h_deg = color.hsv[0] * 360
                weight = color.percentage / 100.0
                
                if 0 <= h_deg < 60 or 300 <= h_deg <= 360:
                    warm_weight += weight
                elif 120 <= h_deg < 240:
                    cool_weight += weight
                else:
                    neutral_weight += weight
        
        total_weight = warm_weight + cool_weight + neutral_weight
        
        if total_weight == 0:
            return {"overall": "neutral", "warm_ratio": 0, "cool_ratio": 0}
        
        warm_ratio = warm_weight / total_weight
        cool_ratio = cool_weight / total_weight
        
        if warm_ratio > cool_ratio * 1.5:
            overall = "warm"
        elif cool_ratio > warm_ratio * 1.5:
            overall = "cool"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "warm_ratio": warm_ratio,
            "cool_ratio": cool_ratio,
            "neutral_ratio": neutral_weight / total_weight
        }
    
    def _calculate_sophistication(self, colors: List[ColorInfo]) -> float:
        """Calculate sophistication score of the palette."""
        if not colors:
            return 0.0
        
        sophistication_score = 0.0
        
        for color in colors:
            if color.hsv:
                h, s, v = color.hsv
                weight = color.percentage / 100.0
                
                # Low saturation, mid-range values are sophisticated
                sat_score = 1.0 - s  # Lower saturation = more sophisticated
                val_score = 1.0 - abs(v - 0.5) * 2  # Mid-range values
                
                color_sophistication = (sat_score + val_score) / 2
                sophistication_score += color_sophistication * weight
        
        return sophistication_score
    
    def _calculate_harmony_score(self, colors: List[ColorInfo]) -> float:
        """Calculate harmony score based on color relationships."""
        if len(colors) < 2:
            return 1.0
        
        harmony_score = 0.0
        comparisons = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                if colors[i].hsv and colors[j].hsv:
                    h1, s1, v1 = colors[i].hsv
                    h2, s2, v2 = colors[j].hsv
                    
                    # Hue harmony (check for common intervals)
                    hue_diff = abs(h1 - h2) * 360
                    hue_diff = min(hue_diff, 360 - hue_diff)
                    
                    # Common harmonious intervals
                    harmonious_intervals = [0, 30, 60, 90, 120, 150, 180]
                    min_diff = min(abs(hue_diff - interval) for interval in harmonious_intervals)
                    hue_harmony = 1.0 - (min_diff / 30.0)  # Normalize to 0-1
                    hue_harmony = max(0.0, hue_harmony)
                    
                    # Saturation harmony (similar saturations work well)
                    sat_diff = abs(s1 - s2)
                    sat_harmony = 1.0 - sat_diff
                    
                    # Value harmony
                    val_diff = abs(v1 - v2)
                    val_harmony = 1.0 - val_diff
                    
                    # Overall harmony for this pair
                    pair_harmony = (hue_harmony * 0.5 + sat_harmony * 0.3 + val_harmony * 0.2)
                    harmony_score += pair_harmony
                    comparisons += 1
        
        return harmony_score / comparisons if comparisons > 0 else 0.0
    
    def _calculate_contrast_level(self, colors: List[ColorInfo]) -> float:
        """Calculate overall contrast level of the palette."""
        if len(colors) < 2:
            return 0.0
        
        total_contrast = 0.0
        comparisons = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                if colors[i].luminance is not None and colors[j].luminance is not None:
                    l1, l2 = colors[i].luminance, colors[j].luminance
                    contrast = abs(l1 - l2)
                    total_contrast += contrast
                    comparisons += 1
        
        return total_contrast / comparisons if comparisons > 0 else 0.0
    
    def _analyze_color_balance(self, colors: List[ColorInfo]) -> Dict[str, Any]:
        """Analyze color balance in the palette."""
        if not colors:
            return {"imbalance_score": 0.0}
        
        # Check distribution across hue ranges
        hue_buckets = [0] * 6  # 6 buckets for hue ranges (60° each)
        total_weight = 0
        
        for color in colors:
            if color.hsv:
                h_deg = color.hsv[0] * 360
                bucket = int(h_deg // 60) % 6
                weight = color.percentage / 100.0
                hue_buckets[bucket] += weight
                total_weight += weight
        
        if total_weight == 0:
            return {"imbalance_score": 0.0}
        
        # Normalize buckets
        normalized_buckets = [b / total_weight for b in hue_buckets]
        
        # Calculate imbalance (standard deviation from uniform distribution)
        uniform_weight = 1.0 / 6
        variance = sum((b - uniform_weight) ** 2 for b in normalized_buckets) / 6
        imbalance_score = math.sqrt(variance) * 6  # Scale to 0-1 range
        
        return {
            "imbalance_score": min(1.0, imbalance_score),
            "hue_distribution": normalized_buckets,
            "dominant_hue_range": normalized_buckets.index(max(normalized_buckets)) * 60
        }
    
    def _determine_seasonal_association(self, colors: List[ColorInfo]) -> str:
        """Determine seasonal association of the palette."""
        if not colors:
            return "neutral"
        
        season_scores = {"spring": 0, "summer": 0, "autumn": 0, "winter": 0}
        
        for color in colors:
            if color.hsv:
                h, s, v = color.hsv
                h_deg = h * 360
                weight = color.percentage / 100.0
                
                # Spring: fresh greens, light colors, pastels
                if 60 <= h_deg <= 120 and s < 0.7 and v > 0.6:
                    season_scores["spring"] += weight
                elif s < 0.4 and v > 0.7:
                    season_scores["spring"] += weight * 0.5
                
                # Summer: bright, saturated colors
                if s > 0.6 and v > 0.6:
                    season_scores["summer"] += weight
                
                # Autumn: oranges, reds, browns
                if 0 <= h_deg <= 60 and s > 0.3:
                    season_scores["autumn"] += weight
                elif 300 <= h_deg <= 360 and s > 0.3:
                    season_scores["autumn"] += weight
                
                # Winter: cool colors, high contrast
                if 180 <= h_deg <= 240:
                    season_scores["winter"] += weight
                elif (s < 0.2 and v < 0.3) or (s < 0.2 and v > 0.8):
                    season_scores["winter"] += weight * 0.5
        
        return max(season_scores, key=season_scores.get)
    
    def _get_style_recommendations(self, colors: List[ColorInfo]) -> List[str]:
        """Get style recommendations based on the palette."""
        recommendations = []
        
        if not colors:
            return recommendations
        
        # Analyze characteristics
        avg_saturation = sum(c.hsv[1] for c in colors if c.hsv) / len(colors)
        avg_value = sum(c.hsv[2] for c in colors if c.hsv) / len(colors)
        
        # Make recommendations based on characteristics
        if avg_saturation > 0.7:
            recommendations.append("Modern and energetic design")
            recommendations.append("Digital interfaces and apps")
        elif avg_saturation < 0.3:
            recommendations.append("Sophisticated and professional")
            recommendations.append("Corporate branding and documentation")
        
        if avg_value > 0.8:
            recommendations.append("Clean and minimalist design")
            recommendations.append("Light and airy layouts")
        elif avg_value < 0.3:
            recommendations.append("Dramatic and bold design")
            recommendations.append("Dark themes and luxury branding")
        
        # Add specific style recommendations
        mood = self._determine_overall_mood(colors)
        if mood == "energetic":
            recommendations.append("Sports and fitness branding")
            recommendations.append("Youth-oriented products")
        elif mood == "sophisticated":
            recommendations.append("Luxury brands and services")
            recommendations.append("Financial and legal services")
        elif mood == "calm":
            recommendations.append("Healthcare and wellness")
            recommendations.append("Meditation and relaxation apps")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _check_contrast_issues(self, colors: List[ColorInfo]) -> List[str]:
        """Check for contrast accessibility issues."""
        issues = []
        
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors[i+1:], i+1):
                if color1.contrast_ratio_white and color2.contrast_ratio_white:
                    # Calculate contrast between the two colors
                    l1 = color1.luminance or 0.5
                    l2 = color2.luminance or 0.5
                    
                    lighter = max(l1, l2)
                    darker = min(l1, l2)
                    contrast_ratio = (lighter + 0.05) / (darker + 0.05)
                    
                    if contrast_ratio < 3.0:  # Poor contrast
                        issues.append(f"Poor contrast between {color1.hex} and {color2.hex} (ratio: {contrast_ratio:.1f})")
        
        return issues[:5]  # Limit to 5 issues
    
    def _check_accessibility_issues(self, colors: List[ColorInfo]) -> List[str]:
        """Check for accessibility issues in the palette."""
        issues = []
        
        # Check if there are colors that might be problematic for color blind users
        red_green_issues = 0
        blue_yellow_issues = 0
        
        for color in colors:
            if color.hsv:
                h_deg = color.hsv[0] * 360
                s = color.hsv[1]
                
                # Red-green color blindness issues
                if ((0 <= h_deg <= 30) or (300 <= h_deg <= 360) or (60 <= h_deg <= 120)) and s > 0.5:
                    red_green_issues += 1
                
                # Blue-yellow color blindness issues
                if (180 <= h_deg <= 240 or 30 <= h_deg <= 90) and s > 0.5:
                    blue_yellow_issues += 1
        
        if red_green_issues >= 2:
            issues.append("May be difficult for users with red-green color blindness")
        
        if blue_yellow_issues >= 2:
            issues.append("May be difficult for users with blue-yellow color blindness")
        
        # Check overall contrast
        if len(colors) > 1:
            avg_contrast = self._calculate_contrast_level(colors)
            if avg_contrast < 0.3:
                issues.append("Overall contrast may be too low for accessibility")
        
        return issues
    
    def _suggest_missing_colors(self, colors: List[ColorInfo]) -> List[ColorInfo]:
        """Suggest colors that might enhance the palette."""
        suggestions = []
        
        if not colors:
            return suggestions
        
        # Analyze what's missing
        hue_coverage = [False] * 12  # 12 segments of 30° each
        
        for color in colors:
            if color.hsv:
                h_deg = color.hsv[0] * 360
                segment = int(h_deg // 30)
                hue_coverage[segment] = True
        
        # Find missing hue ranges and suggest colors
        missing_segments = [i for i, covered in enumerate(hue_coverage) if not covered]
        
        if missing_segments and len(missing_segments) <= 6:  # Don't suggest too many
            base_color = colors[0]  # Use dominant color as reference
            base_s, base_v = base_color.hsv[1:] if base_color.hsv else (0.5, 0.5)
            
            for segment in missing_segments[:3]:  # Limit to 3 suggestions
                suggested_hue = (segment * 30 + 15) / 360  # Middle of segment
                suggested_color = self._create_color_info_hsv(suggested_hue, base_s, base_v)
                suggestions.append(suggested_color)
        
        return suggestions
