# 🎨 PyColour Extract 2.0

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/badge/release-v2.0.0-orange.svg)](https://github.com/Kynlos/Python-Colour-Info-Extractor)

**Advanced color extraction and analysis tool with machine learning capabilities, comprehensive export options, and professional design insights.**

## ✨ What's New in 2.0

This is a complete rewrite and massive upgrade from the original tool, featuring:

### 🚀 Core Improvements
- **Multiple extraction algorithms**: K-means, DBSCAN, Median Cut, Octree, Histogram analysis
- **Advanced color analysis**: Harmony detection, accessibility scoring, emotional analysis
- **Professional insights**: Brand personality analysis, color temperature, vibrance metrics
- **Modern architecture**: Modular design with proper packaging and type hints

### 🎯 New Features
- **Rich CLI interface** with beautiful output and progress tracking
- **20+ export formats**: JSON, CSS, SCSS, ASE, ACO, Swift, Kotlin, Python, and more
- **Batch processing** with parallel execution for multiple images
- **Color comparison** between different images
- **Color suggestions** based on harmony rules and accessibility
- **Comprehensive analysis**: Color blindness simulation, contrast ratios, luminance

### 🔧 Professional Tools
- **Color clustering** with variance analysis
- **Harmony detection**: Complementary, triadic, analogous, split-complementary
- **Accessibility analysis** with WCAG compliance checking
- **Brand personality insights** based on color psychology
- **Emotional associations** and temperature analysis

## 📦 Installation

### Install from GitHub

```bash
# Clone and install directly from GitHub
git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git
cd Python-Colour-Info-Extractor
pip install -e .

# Install with all optional dependencies
pip install -e ".[gui,export,dev]"

# Or install directly from GitHub URL
pip install git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git
```

### Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🚀 Quick Start

### Command Line Interface

```bash
# Basic color extraction
pycolour-extract image.jpg

# Advanced analysis with multiple formats
pycolour-extract image.jpg --algorithm kmeans --export-formats json css scss html

# Full analysis with branding insights
pycolour-extract image.jpg --analyze-emotion --analyze-brand --analyze-accessibility

# Batch process a directory
pycolour-extract batch ./images --recursive --export-formats json html

# Compare color palettes
pycolour-extract compare image1.jpg image2.jpg image3.jpg

# Generate color suggestions
pycolour-extract suggest "#FF0000" "#00FF00" --suggestion-type harmony
```

### Python API

```python
from pycolour_extract import ColorExtractor, ColorAnalyzer

# Extract colors with advanced analysis
extractor = ColorExtractor(algorithm="kmeans", max_colors=16)
result = extractor.extract_colors("image.jpg")

# Access color data
colors = result.color_data.colors
dominant = result.color_data.dominant_color
harmonies = result.color_data.harmonies

# Perform additional analysis
analyzer = ColorAnalyzer()
emotions = analyzer.calculate_color_emotion(colors)
brand_personality = analyzer.analyze_brand_personality(colors)

# Export to various formats
from pycolour_extract.exporters import FormatExporter
from pycolour_extract.models.color_data import ExportFormat

exporter = FormatExporter()
exporter.export(result, ExportFormat.CSS, "colors.css")
exporter.export(result, ExportFormat.JSON, "analysis.json")
```

## 🎨 Extraction Algorithms

### K-Means Clustering (Default)
- **Best for**: General purpose, balanced results
- **Pros**: Fast, reliable, good color separation
- **Use case**: Most images, product photography

### DBSCAN Clustering
- **Best for**: Images with noise, irregular color distributions
- **Pros**: Handles outliers well, finds natural groupings
- **Use case**: Artistic images, complex backgrounds

### Median Cut
- **Best for**: Preserving important colors, uniform distribution
- **Pros**: Perceptually accurate, maintains color relationships
- **Use case**: Logos, graphics with important color relationships

### Octree Quantization
- **Best for**: Fast processing, web applications
- **Pros**: Very fast, good for real-time applications
- **Use case**: Batch processing, web services

### Histogram Analysis
- **Best for**: Scientific analysis, peak detection
- **Pros**: Finds all significant colors, detailed analysis
- **Use case**: Research, detailed color studies

## 📊 Analysis Features

### Color Harmony Detection
- **Complementary**: Colors opposite on the color wheel
- **Triadic**: Three evenly spaced colors
- **Analogous**: Adjacent colors on the wheel
- **Split-Complementary**: Base color + two colors adjacent to complement
- **Tetradic**: Four colors forming a rectangle

### Accessibility Analysis
- **WCAG Compliance**: Automatic contrast ratio checking
- **Color Blindness Simulation**: Protanopia, Deuteranopia, Tritanopia
- **Accessibility Scoring**: Overall accessibility rating
- **Recommendations**: Suggestions for improvement

### Brand Personality Analysis
Analyzes colors to determine brand personality traits:
- **Modern vs Traditional**
- **Professional vs Casual** 
- **Bold vs Subtle**
- **Masculine vs Feminine**
- **Youthful vs Mature**

### Emotional Associations
Calculates emotional impact based on color psychology:
- **Energy & Excitement**
- **Calmness & Serenity**
- **Warmth & Comfort**
- **Sophistication & Luxury**
- **Trustworthiness & Reliability**
- **Creativity & Innovation**

## 🎯 Export Formats

### Design & Development
- **CSS**: Custom properties and utility classes
- **SCSS/LESS/Stylus**: Variables and mixins
- **ASE**: Adobe Swatch Exchange
- **ACO**: Adobe Color format
- **GPL**: GIMP Palette

### Programming Languages
- **JavaScript**: ES6 modules and CommonJS
- **Python**: Constants and dictionaries
- **Swift**: UIColor extensions
- **Kotlin**: Color constants
- **Java**: Color class definitions

### Data Formats
- **JSON**: Complete analysis data
- **CSV/TSV**: Tabular color data
- **XML**: Structured analysis
- **YAML**: Human-readable format

### Visual Formats
- **HTML**: Interactive color report
- **SVG**: Scalable palette graphics
- **PNG**: Palette images
- **PDF**: Professional reports

## 🔧 Advanced Usage

### Custom Analysis Pipeline

```python
from pycolour_extract import ColorExtractor, ColorAnalyzer
from pycolour_extract.models.color_data import ExportFormat

# Configure extractor
extractor = ColorExtractor(
    algorithm="kmeans",
    max_colors=20
)

# Extract with clustering
result = extractor.extract_colors(
    "image.jpg",
    n_clusters=5,
    cluster_colors=True
)

# Advanced analysis
analyzer = ColorAnalyzer()

# Color relationships
relationships = analyzer.analyze_color_relationships(result.color_data.colors)

# Color blindness impact
cb_analysis = analyzer.calculate_color_blindness_impact(result.color_data.colors)

# Generate suggestions
suggestions = analyzer.generate_color_suggestions(
    result.color_data.colors[:3],
    suggestion_type="accessibility"
)

print(f"Found {len(result.color_data.colors)} colors")
print(f"Dominant color: {result.color_data.dominant_color.hex}")
print(f"Color temperature: {result.color_data.color_temperature}K")
print(f"Accessibility score: {result.color_data.accessibility_score}/1.0")
```

### Batch Processing with Custom Settings

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pycolour_extract import ColorExtractor
from pycolour_extract.exporters import FormatExporter

def process_image(image_path):
    extractor = ColorExtractor(algorithm="kmeans", max_colors=12)
    result = extractor.extract_colors(image_path)
    
    # Export multiple formats
    exporter = FormatExporter()
    output_dir = image_path.parent / f"{image_path.stem}_analysis"
    output_dir.mkdir(exist_ok=True)
    
    exporter.export(result, ExportFormat.JSON, output_dir / "colors.json")
    exporter.export(result, ExportFormat.CSS, output_dir / "colors.css")
    exporter.export(result, ExportFormat.HTML, output_dir / "report.html")
    
    return result

# Process multiple images in parallel
image_dir = Path("./images")
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_files))

print(f"Processed {len(results)} images")
```

## 🎨 Color Theory Integration

PyColour Extract 2.0 incorporates advanced color theory:

### Perceptual Color Spaces
- **RGB**: Standard display colors
- **HSV**: Hue, Saturation, Value
- **HSL**: Hue, Saturation, Lightness  
- **LAB**: Perceptually uniform color space
- **LCH**: Lightness, Chroma, Hue
- **XYZ**: CIE color space

### Color Difference Metrics
- **Delta E CIE2000**: Most accurate color difference
- **Delta E CIE1994**: Industry standard
- **Delta E CIE1976**: Simple Euclidean distance
- **RGB Euclidean**: Fast approximation

### Professional Features
- **Color Temperature**: Warm/cool analysis in Kelvin
- **Vibrance**: Saturation intensity measurement
- **Luminance**: Perceptual brightness
- **Contrast Ratios**: WCAG accessibility compliance

## 🔧 Configuration

### CLI Configuration
Create a `.pycolour-extract.json` config file:

```json
{
  "default_algorithm": "kmeans",
  "max_colors": 16,
  "export_formats": ["json", "css", "html"],
  "analyze_emotion": true,
  "analyze_brand": true,
  "analyze_accessibility": true,
  "output_template": "{filename}_analysis"
}
```

### Python Configuration

```python
from pycolour_extract.config import Config

config = Config(
    algorithm="dbscan",
    max_colors=20,
    enable_clustering=True,
    cluster_count=5,
    analyze_harmony=True,
    calculate_emotions=True
)

extractor = ColorExtractor.from_config(config)
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=pycolour_extract --cov-report=html

# Run specific test types
pytest tests/test_extraction.py -v
pytest tests/test_analysis.py -v
pytest tests/test_export.py -v
```

## 📈 Performance

### Benchmarks
- **Small images** (800x600): ~0.5-2 seconds
- **Medium images** (1920x1080): ~2-8 seconds  
- **Large images** (4K+): ~8-30 seconds

### Optimization Tips
- Use `algorithm="octree"` for fastest processing
- Reduce `max_colors` for better performance
- Enable parallel processing for batch operations
- Use `quiet=True` to disable progress output

## 🎯 Use Cases

### Design & Branding
- **Brand color extraction** from logos and marketing materials
- **Mood board creation** with automatic color harmonies
- **Accessibility auditing** for web and print design
- **Color trend analysis** across image collections

### Development
- **Theme generation** for applications and websites
- **Asset optimization** with dominant color backgrounds
- **Color-based image tagging** and categorization
- **Automated style guide creation**

### Research & Analysis
- **Color psychology studies** with emotional analysis
- **Cultural color analysis** across different regions
- **Product color research** for market analysis
- **Scientific color measurement** and documentation

### E-commerce
- **Product color extraction** for search and filtering
- **Automatic color tagging** for inventory systems
- **Color-based recommendations** for similar products
- **Brand compliance checking** across product lines

## 🆚 Comparison with Original

| Feature | Original (v1) | PyColour Extract 2.0 |
|---------|----------|---------------------|
| **Installation** | Single Python file | Professional package from GitHub |
| **Algorithms** | 1 (basic pixel analysis) | 5 (advanced ML: K-means, DBSCAN, etc.) |
| **Export Formats** | 2 (txt, png) | 20+ (CSS, SCSS, JSON, ASE, PDF, etc.) |
| **Analysis** | Basic RGB stats | Professional color theory & psychology |
| **Performance** | Single-threaded | Multi-threaded with parallel processing |
| **Interface** | Drag & drop only | Rich CLI + Python API |
| **Color Theory** | Basic RGB/HSV | Advanced (LAB, LCH, Delta E, etc.) |
| **Accessibility** | None | Full WCAG analysis & color blindness |
| **Batch Processing** | None | Directory processing with progress |
| **Documentation** | Basic README | Comprehensive docs & examples |
| **Code Quality** | Script-style | Modern packaging with type hints |

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git
cd Python-Colour-Info-Extractor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install

# Run tests
pytest
```

### First-Time Setup

If you're new to the project:

```bash
# 1. Make sure you have Python 3.8+ and Git installed
python --version
git --version

# 2. Clone and enter the project directory
git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git
cd Python-Colour-Info-Extractor

# 3. Install the package
pip install -e .

# 4. Test the installation
pycolour-extract --version

# 5. Try it with an image
pycolour-extract path/to/your/image.jpg
```

### Project Structure

```
src/pycolour_extract/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── core/                    # Core functionality
│   ├── color_extractor.py   # Color extraction algorithms
│   ├── color_analyzer.py    # Color analysis and theory
│   └── palette_generator.py # Palette generation
├── models/                  # Data models
│   └── color_data.py        # Color data structures
├── exporters/              # Export functionality
│   └── format_exporter.py   # Multi-format export
├── gui/                    # GUI components (optional)
│   └── main_window.py       # Main GUI application
└── utils/                  # Utility functions
    ├── color_utils.py       # Color manipulation utilities
    └── image_utils.py       # Image processing utilities
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- **New algorithms**: Additional color extraction methods
- **Export formats**: Support for more design tools
- **Analysis features**: Advanced color theory implementations
- **Performance**: Optimization and GPU acceleration
- **Documentation**: Tutorials and examples
- **GUI improvements**: Enhanced user interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original PyColour Extract**: Foundation and inspiration
- **scikit-learn**: Machine learning algorithms
- **Pillow**: Image processing capabilities
- **colormath**: Color space conversions and theory
- **Rich**: Beautiful command-line interface
- **Color theory research**: Academic papers and industry standards

## 📚 Resources

### Color Theory
- [Color Theory Basics](https://www.adobe.com/creativecloud/design/discover/color-theory.html)
- [WCAG Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/)
- [Color Psychology in Design](https://www.canva.com/colors/color-psychology/)

### Technical Documentation
- [Delta E Color Difference](https://en.wikipedia.org/wiki/Color_difference)
- [CIE Color Spaces](https://en.wikipedia.org/wiki/CIE_1931_color_space)
- [Color Harmony Theory](https://www.sessions.edu/color-calculator/)

## 🔮 Roadmap

### Version 2.1
- **GUI Application**: Desktop application with real-time preview
- **Web Interface**: Browser-based color extraction tool
- **API Service**: REST API for integration with other tools
- **Mobile App**: Color extraction on mobile devices

### Version 2.2
- **AI-Powered Analysis**: Machine learning for color trend prediction
- **Cloud Processing**: Scalable cloud-based extraction
- **Real-time Processing**: Webcam and video color extraction
- **Advanced Clustering**: Deep learning-based color grouping

### Version 3.0
- **Multi-modal Analysis**: Text, audio, and color correlation
- **Augmented Reality**: AR color extraction and overlay
- **Enterprise Features**: Team collaboration and brand management
- **API Integrations**: Direct integration with design tools

---

**PyColour Extract 2.0** - Transform your color analysis workflow with professional-grade tools and insights.

For support, feature requests, or contributions, visit our [GitHub repository](https://github.com/Kynlos/Python-Colour-Info-Extractor).

## 📥 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git
   cd Python-Colour-Info-Extractor
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

3. **Test with an image:**
   ```bash
   pycolour-extract your-image.jpg
   ```

4. **Explore advanced features:**
   ```bash
   pycolour-extract --help
   ```

The tool will create an analysis directory with your results in multiple formats!
