"""Advanced color extraction engine with multiple algorithms."""

import time
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image, ExifTags
import cv2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import colorsys
import webcolors
from colormath.color_objects import sRGBColor, LabColor, LCHabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from ..models.color_data import (
    ColorData, ColorInfo, ImageMetadata, ColorCluster, 
    ColorHarmony, AnalysisResult, PaletteData
)


class ColorExtractor:
    """Advanced color extraction with multiple algorithms and analysis methods."""
    
    def __init__(self, algorithm: str = "kmeans", max_colors: int = 256):
        """
        Initialize the color extractor.
        
        Args:
            algorithm: Extraction algorithm ('kmeans', 'dbscan', 'median_cut', 'octree')
            max_colors: Maximum number of colors to extract
        """
        self.algorithm = algorithm
        self.max_colors = max_colors
        self.supported_algorithms = ['kmeans', 'dbscan', 'median_cut', 'octree', 'histogram']
        
        if algorithm not in self.supported_algorithms:
            raise ValueError(f"Algorithm must be one of {self.supported_algorithms}")
    
    def extract_colors(
        self, 
        image_path: Union[str, Path], 
        **kwargs
    ) -> AnalysisResult:
        """
        Extract colors from an image using the specified algorithm.
        
        Args:
            image_path: Path to the image file
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            AnalysisResult containing complete color analysis
        """
        start_time = time.time()
        
        # Load and prepare image
        image = Image.open(image_path).convert('RGB')
        image_metadata = self._extract_metadata(image_path, image)
        
        # Extract colors using selected algorithm
        if self.algorithm == 'kmeans':
            colors = self._extract_kmeans(image, **kwargs)
        elif self.algorithm == 'dbscan':
            colors = self._extract_dbscan(image, **kwargs)
        elif self.algorithm == 'median_cut':
            colors = self._extract_median_cut(image, **kwargs)
        elif self.algorithm == 'octree':
            colors = self._extract_octree(image, **kwargs)
        else:  # histogram
            colors = self._extract_histogram(image, **kwargs)
        
        # Calculate additional color information
        colors = self._enhance_color_info(colors, image)
        
        # Find dominant and average colors
        dominant_color = max(colors, key=lambda x: x.frequency)
        average_color = self._calculate_average_color(image)
        
        # Perform clustering analysis
        clusters = self._perform_clustering(colors, n_clusters=kwargs.get('n_clusters', 5))
        
        # Detect color harmonies
        harmonies = self._detect_harmonies(colors)
        
        # Calculate additional metrics
        accessibility_score = self._calculate_accessibility_score(colors)
        color_temperature = self._calculate_color_temperature(colors)
        vibrance = self._calculate_vibrance(colors)
        saturation_dist = self._analyze_saturation_distribution(colors)
        
        color_data = ColorData(
            image_metadata=image_metadata,
            colors=colors,
            unique_color_count=len(colors),
            dominant_color=dominant_color,
            average_color=average_color,
            clusters=clusters,
            harmonies=harmonies,
            accessibility_score=accessibility_score,
            color_temperature=color_temperature,
            vibrance=vibrance,
            saturation_distribution=saturation_dist
        )
        
        # Generate palettes
        palettes = self._generate_palettes(colors, image_path)
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            color_data=color_data,
            palettes=palettes,
            processing_time=processing_time,
            algorithm_used=self.algorithm,
            settings=kwargs
        )
    
    def _extract_metadata(self, image_path: Union[str, Path], image: Image.Image) -> ImageMetadata:
        """Extract image metadata."""
        path_obj = Path(image_path)
        stat = path_obj.stat()
        
        # Try to get EXIF data
        created_at = None
        modified_at = None
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        if tag_name == 'DateTime':
                            created_at = str(value)
        except:
            pass
        
        modified_at = time.ctime(stat.st_mtime)
        
        return ImageMetadata(
            path=str(image_path),
            filename=path_obj.name,
            size=image.size,
            format=image.format or 'Unknown',
            mode=image.mode,
            total_pixels=image.size[0] * image.size[1],
            file_size=stat.st_size,
            created_at=created_at,
            modified_at=modified_at
        )
    
    def _extract_kmeans(self, image: Image.Image, n_clusters: int = 8, **kwargs) -> List[ColorInfo]:
        """Extract colors using K-means clustering."""
        # Convert to numpy array
        pixels = np.array(image).reshape(-1, 3)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=min(n_clusters, self.max_colors), random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        colors = []
        total_pixels = len(pixels)
        
        for i, center in enumerate(kmeans.cluster_centers_):
            rgb = tuple(map(int, center))
            frequency = np.sum(labels == i)
            percentage = (frequency / total_pixels) * 100
            
            color_info = ColorInfo(
                rgb=rgb,
                hex=self._rgb_to_hex(rgb),
                frequency=frequency,
                percentage=percentage
            )
            colors.append(color_info)
        
        return sorted(colors, key=lambda x: x.frequency, reverse=True)
    
    def _extract_dbscan(self, image: Image.Image, eps: float = 10, min_samples: int = 50, **kwargs) -> List[ColorInfo]:
        """Extract colors using DBSCAN clustering."""
        pixels = np.array(image).reshape(-1, 3)
        
        # Standardize the data
        scaler = StandardScaler()
        pixels_scaled = scaler.fit_transform(pixels)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pixels_scaled)
        
        colors = []
        total_pixels = len(pixels)
        
        # Get unique labels (excluding noise points labeled as -1)
        unique_labels = np.unique(labels[labels != -1])
        
        for label in unique_labels:
            cluster_pixels = pixels[labels == label]
            rgb = tuple(map(int, np.mean(cluster_pixels, axis=0)))
            frequency = len(cluster_pixels)
            percentage = (frequency / total_pixels) * 100
            
            color_info = ColorInfo(
                rgb=rgb,
                hex=self._rgb_to_hex(rgb),
                frequency=frequency,
                percentage=percentage
            )
            colors.append(color_info)
        
        return sorted(colors, key=lambda x: x.frequency, reverse=True)[:self.max_colors]
    
    def _extract_median_cut(self, image: Image.Image, **kwargs) -> List[ColorInfo]:
        """Extract colors using median cut algorithm."""
        # Use PIL's quantize method which implements median cut
        quantized = image.quantize(colors=min(self.max_colors, 256))
        palette = quantized.getpalette()
        
        colors = []
        # Convert palette to RGB tuples
        for i in range(0, len(palette), 3):
            rgb = (palette[i], palette[i+1], palette[i+2])
            if rgb != (0, 0, 0):  # Skip pure black which might be padding
                color_info = ColorInfo(
                    rgb=rgb,
                    hex=self._rgb_to_hex(rgb),
                    frequency=1,  # Will be updated later
                    percentage=0.0
                )
                colors.append(color_info)
        
        # Calculate actual frequencies
        return self._calculate_frequencies(colors, image)
    
    def _extract_octree(self, image: Image.Image, **kwargs) -> List[ColorInfo]:
        """Extract colors using octree quantization."""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply quantization
        data = cv_image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = min(self.max_colors, 16)
        
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        colors = []
        total_pixels = len(data)
        
        for i, center in enumerate(centers):
            # Convert BGR back to RGB
            rgb = tuple(map(int, center[::-1]))
            frequency = np.sum(labels == i)
            percentage = (frequency / total_pixels) * 100
            
            color_info = ColorInfo(
                rgb=rgb,
                hex=self._rgb_to_hex(rgb),
                frequency=frequency,
                percentage=percentage
            )
            colors.append(color_info)
        
        return sorted(colors, key=lambda x: x.frequency, reverse=True)
    
    def _extract_histogram(self, image: Image.Image, bins: int = 32, **kwargs) -> List[ColorInfo]:
        """Extract colors using histogram analysis."""
        # Convert to HSV for better color separation
        hsv_image = image.convert('HSV')
        pixels = np.array(hsv_image)
        
        # Create 3D histogram
        hist, edges = np.histogramdd(
            pixels.reshape(-1, 3), 
            bins=[bins, bins, bins], 
            range=[(0, 360), (0, 255), (0, 255)]
        )
        
        colors = []
        total_pixels = pixels.shape[0] * pixels.shape[1]
        
        # Find peaks in histogram
        threshold = total_pixels * 0.001  # At least 0.1% of pixels
        peaks = np.where(hist > threshold)
        
        for h, s, v in zip(*peaks):
            # Convert bin indices back to HSV values
            hue = edges[0][h] + (edges[0][1] - edges[0][0]) / 2
            sat = edges[1][s] + (edges[1][1] - edges[1][0]) / 2
            val = edges[2][v] + (edges[2][1] - edges[2][0]) / 2
            
            # Convert to RGB
            rgb = tuple(map(int, colorsys.hsv_to_rgb(hue/360, sat/255, val/255)))
            rgb = tuple(c * 255 for c in rgb)
            rgb = tuple(map(int, rgb))
            
            frequency = int(hist[h, s, v])
            percentage = (frequency / total_pixels) * 100
            
            color_info = ColorInfo(
                rgb=rgb,
                hex=self._rgb_to_hex(rgb),
                frequency=frequency,
                percentage=percentage
            )
            colors.append(color_info)
        
        return sorted(colors, key=lambda x: x.frequency, reverse=True)[:self.max_colors]
    
    def _enhance_color_info(self, colors: List[ColorInfo], image: Image.Image) -> List[ColorInfo]:
        """Enhance color information with additional color space conversions and metrics."""
        for color in colors:
            r, g, b = color.rgb
            
            # Convert to other color spaces
            color.hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            color.hsl = colorsys.rgb_to_hls(r/255, g/255, b/255)
            
            # Convert to LAB
            srgb_color = sRGBColor(r/255, g/255, b/255)
            lab_color = convert_color(srgb_color, LabColor)
            color.lab = (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)
            
            # Calculate CMYK
            color.cmyk = self._rgb_to_cmyk(r, g, b)
            
            # Calculate luminance
            color.luminance = self._calculate_luminance(r, g, b)
            
            # Calculate contrast ratios
            color.contrast_ratio_white = self._calculate_contrast_ratio((r, g, b), (255, 255, 255))
            color.contrast_ratio_black = self._calculate_contrast_ratio((r, g, b), (0, 0, 0))
            
            # Try to find color name
            try:
                color.name = webcolors.rgb_to_name((r, g, b))
            except ValueError:
                # Find closest named color
                color.name = self._find_closest_color_name((r, g, b))
        
        return colors
    
    def _calculate_frequencies(self, colors: List[ColorInfo], image: Image.Image) -> List[ColorInfo]:
        """Calculate actual color frequencies in the image."""
        pixels = np.array(image)
        total_pixels = pixels.shape[0] * pixels.shape[1]
        
        for color in colors:
            # Count pixels matching this color (with some tolerance)
            mask = np.all(np.abs(pixels - np.array(color.rgb)) < 5, axis=2)
            frequency = np.sum(mask)
            color.frequency = frequency
            color.percentage = (frequency / total_pixels) * 100
        
        return sorted(colors, key=lambda x: x.frequency, reverse=True)
    
    def _calculate_average_color(self, image: Image.Image) -> ColorInfo:
        """Calculate the average color of the image."""
        # Resize to 1x1 to get average color
        avg_color = image.resize((1, 1)).getpixel((0, 0))
        
        color_info = ColorInfo(
            rgb=avg_color,
            hex=self._rgb_to_hex(avg_color),
            frequency=0,
            percentage=0.0
        )
        
        # Enhance with additional info
        enhanced = self._enhance_color_info([color_info], image)
        return enhanced[0]
    
    def _perform_clustering(self, colors: List[ColorInfo], n_clusters: int = 5) -> List[ColorCluster]:
        """Perform clustering analysis on extracted colors."""
        if len(colors) < n_clusters:
            n_clusters = len(colors)
        
        # Prepare data for clustering
        color_data = np.array([color.rgb for color in colors])
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(color_data)
        
        clusters = []
        for i in range(n_clusters):
            cluster_colors = [colors[j] for j in range(len(colors)) if labels[j] == i]
            if cluster_colors:
                centroid_rgb = tuple(map(int, kmeans.cluster_centers_[i]))
                centroid = ColorInfo(
                    rgb=centroid_rgb,
                    hex=self._rgb_to_hex(centroid_rgb),
                    frequency=sum(c.frequency for c in cluster_colors),
                    percentage=sum(c.percentage for c in cluster_colors)
                )
                
                # Calculate cluster variance
                variance = np.var([np.linalg.norm(np.array(c.rgb) - np.array(centroid_rgb)) 
                                 for c in cluster_colors])
                
                cluster = ColorCluster(
                    cluster_id=i,
                    centroid=centroid,
                    colors=cluster_colors,
                    size=len(cluster_colors),
                    variance=float(variance)
                )
                clusters.append(cluster)
        
        return sorted(clusters, key=lambda x: x.size, reverse=True)
    
    def _detect_harmonies(self, colors: List[ColorInfo]) -> List[ColorHarmony]:
        """Detect color harmonies in the extracted colors."""
        harmonies = []
        
        # Take top colors for harmony analysis
        top_colors = colors[:min(10, len(colors))]
        
        for base_color in top_colors[:3]:  # Check top 3 colors as potential base
            h, s, v = base_color.hsv
            h_deg = h * 360
            
            # Complementary harmony (180° apart)
            comp_h = (h_deg + 180) % 360
            comp_colors = self._find_colors_near_hue(colors, comp_h, tolerance=15)
            if comp_colors:
                harmony = ColorHarmony(
                    harmony_type="complementary",
                    base_color=base_color,
                    harmony_colors=comp_colors,
                    confidence=min(1.0, len(comp_colors) * 0.3)
                )
                harmonies.append(harmony)
            
            # Triadic harmony (120° apart)
            tri_h1 = (h_deg + 120) % 360
            tri_h2 = (h_deg + 240) % 360
            tri_colors1 = self._find_colors_near_hue(colors, tri_h1, tolerance=20)
            tri_colors2 = self._find_colors_near_hue(colors, tri_h2, tolerance=20)
            if tri_colors1 and tri_colors2:
                harmony = ColorHarmony(
                    harmony_type="triadic",
                    base_color=base_color,
                    harmony_colors=tri_colors1 + tri_colors2,
                    confidence=min(1.0, (len(tri_colors1) + len(tri_colors2)) * 0.2)
                )
                harmonies.append(harmony)
            
            # Analogous harmony (30° range)
            analog_colors = self._find_colors_near_hue(colors, h_deg, tolerance=30, exclude_base=True)
            if analog_colors and len(analog_colors) >= 2:
                harmony = ColorHarmony(
                    harmony_type="analogous",
                    base_color=base_color,
                    harmony_colors=analog_colors[:4],
                    confidence=min(1.0, len(analog_colors) * 0.25)
                )
                harmonies.append(harmony)
        
        return sorted(harmonies, key=lambda x: x.confidence, reverse=True)
    
    def _find_colors_near_hue(self, colors: List[ColorInfo], target_hue: float, 
                             tolerance: float = 15, exclude_base: bool = False) -> List[ColorInfo]:
        """Find colors near a specific hue."""
        near_colors = []
        
        for color in colors:
            if exclude_base and color.hsv[0] * 360 == target_hue:
                continue
                
            h = color.hsv[0] * 360
            hue_diff = min(abs(h - target_hue), 360 - abs(h - target_hue))
            
            if hue_diff <= tolerance:
                near_colors.append(color)
        
        return sorted(near_colors, key=lambda x: x.frequency, reverse=True)
    
    def _calculate_accessibility_score(self, colors: List[ColorInfo]) -> float:
        """Calculate accessibility score based on contrast ratios."""
        if len(colors) < 2:
            return 0.0
        
        total_score = 0.0
        comparisons = 0
        
        for i, color1 in enumerate(colors[:5]):  # Check top 5 colors
            for color2 in colors[i+1:6]:
                contrast = self._calculate_contrast_ratio(color1.rgb, color2.rgb)
                # WCAG AA standard is 4.5:1 for normal text
                score = min(1.0, contrast / 4.5)
                total_score += score
                comparisons += 1
        
        return total_score / comparisons if comparisons > 0 else 0.0
    
    def _calculate_color_temperature(self, colors: List[ColorInfo]) -> float:
        """Calculate average color temperature in Kelvin."""
        total_temp = 0.0
        total_weight = 0.0
        
        for color in colors[:10]:  # Use top 10 colors
            r, g, b = [c / 255.0 for c in color.rgb]
            
            # Convert RGB to XYZ to calculate temperature
            temp = self._rgb_to_color_temperature(r, g, b)
            if temp:
                total_temp += temp * color.percentage
                total_weight += color.percentage
        
        return total_temp / total_weight if total_weight > 0 else 6500.0  # Default daylight
    
    def _calculate_vibrance(self, colors: List[ColorInfo]) -> float:
        """Calculate overall vibrance/saturation of the color palette."""
        total_saturation = 0.0
        total_weight = 0.0
        
        for color in colors:
            saturation = color.hsv[1]  # HSV saturation
            total_saturation += saturation * color.percentage
            total_weight += color.percentage
        
        return total_saturation / total_weight if total_weight > 0 else 0.0
    
    def _analyze_saturation_distribution(self, colors: List[ColorInfo]) -> Dict[str, float]:
        """Analyze the distribution of saturation levels."""
        low_sat = medium_sat = high_sat = 0.0
        
        for color in colors:
            sat = color.hsv[1]
            weight = color.percentage
            
            if sat < 0.3:
                low_sat += weight
            elif sat < 0.7:
                medium_sat += weight
            else:
                high_sat += weight
        
        return {
            "low_saturation": low_sat,
            "medium_saturation": medium_sat,
            "high_saturation": high_sat
        }
    
    def _generate_palettes(self, colors: List[ColorInfo], source_image: str) -> List[PaletteData]:
        """Generate different types of color palettes."""
        palettes = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Dominant palette (top colors by frequency)
        dominant_colors = colors[:8]
        palettes.append(PaletteData(
            name="Dominant Colors",
            colors=dominant_colors,
            palette_type="dominant",
            source_image=str(source_image),
            created_at=timestamp
        ))
        
        # Vibrant palette (high saturation colors)
        vibrant_colors = [c for c in colors if c.hsv[1] > 0.5][:6]
        if vibrant_colors:
            palettes.append(PaletteData(
                name="Vibrant Colors",
                colors=vibrant_colors,
                palette_type="vibrant",
                source_image=str(source_image),
                created_at=timestamp
            ))
        
        # Muted palette (low saturation colors)
        muted_colors = [c for c in colors if c.hsv[1] < 0.3 and c.hsv[2] > 0.3][:6]
        if muted_colors:
            palettes.append(PaletteData(
                name="Muted Colors",
                colors=muted_colors,
                palette_type="muted",
                source_image=str(source_image),
                created_at=timestamp
            ))
        
        # Dark palette
        dark_colors = [c for c in colors if c.hsv[2] < 0.4][:6]
        if dark_colors:
            palettes.append(PaletteData(
                name="Dark Colors",
                colors=dark_colors,
                palette_type="dark",
                source_image=str(source_image),
                created_at=timestamp
            ))
        
        # Light palette
        light_colors = [c for c in colors if c.hsv[2] > 0.8][:6]
        if light_colors:
            palettes.append(PaletteData(
                name="Light Colors",
                colors=light_colors,
                palette_type="light",
                source_image=str(source_image),
                created_at=timestamp
            ))
        
        return palettes
    
    # Utility methods
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex."""
        return "#{:02x}{:02x}{:02x}".format(*rgb)
    
    def _rgb_to_cmyk(self, r: int, g: int, b: int) -> Tuple[float, float, float, float]:
        """Convert RGB to CMYK."""
        r, g, b = r/255.0, g/255.0, b/255.0
        k = 1 - max(r, g, b)
        if k == 1:
            return (0.0, 0.0, 0.0, 1.0)
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
        return (c, m, y, k)
    
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
    
    def _calculate_contrast_ratio(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate contrast ratio between two colors."""
        l1 = self._calculate_luminance(*color1)
        l2 = self._calculate_luminance(*color2)
        
        lighter = max(l1, l2)
        darker = min(l1, l2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def _find_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Find the closest named color."""
        min_diff = float('inf')
        closest_name = "Unknown"
        
        try:
            # Try to get the color name map from webcolors
            import webcolors._conversion
            color_names = webcolors._conversion._get_hex_to_name_map('css3')
            
            for hex_color, name in color_names.items():
                try:
                    named_rgb = webcolors.hex_to_rgb(hex_color)
                    diff = sum((a - b) ** 2 for a, b in zip(rgb, named_rgb))
                    if diff < min_diff:
                        min_diff = diff
                        closest_name = name
                except:
                    continue
                    
        except (ImportError, AttributeError):
            # Fallback: create a basic color map
            basic_colors = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'red': (255, 0, 0),
                'green': (0, 128, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255),
                'silver': (192, 192, 192),
                'gray': (128, 128, 128),
                'maroon': (128, 0, 0),
                'olive': (128, 128, 0),
                'lime': (0, 255, 0),
                'aqua': (0, 255, 255),
                'teal': (0, 128, 128),
                'navy': (0, 0, 128),
                'fuchsia': (255, 0, 255),
                'purple': (128, 0, 128)
            }
            
            for name, color_rgb in basic_colors.items():
                diff = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
                if diff < min_diff:
                    min_diff = diff
                    closest_name = name
        
        return closest_name
    
    def _rgb_to_color_temperature(self, r: float, g: float, b: float) -> Optional[float]:
        """Estimate color temperature from RGB values."""
        # This is an approximation - more accurate methods require more complex calculations
        try:
            # Convert to XYZ color space first (simplified)
            x = r * 0.4124 + g * 0.3576 + b * 0.1805
            y = r * 0.2126 + g * 0.7152 + b * 0.0722
            z = r * 0.0193 + g * 0.1192 + b * 0.9505
            
            if x + y + z == 0:
                return None
            
            # Calculate chromaticity coordinates
            x_chrom = x / (x + y + z)
            y_chrom = y / (x + y + z)
            
            # Estimate temperature using McCamy's approximation
            n = (x_chrom - 0.3320) / (0.1858 - y_chrom)
            temp = 437 * n**3 + 3601 * n**2 + 6861 * n + 5517
            
            # Clamp to reasonable range
            return max(1000, min(25000, temp))
        except:
            return None
