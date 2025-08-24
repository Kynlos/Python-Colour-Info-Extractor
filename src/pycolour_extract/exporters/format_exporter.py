"""Export analysis results to various formats."""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Union
import struct
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..models.color_data import AnalysisResult, ExportFormat, ColorInfo


class FormatExporter:
    """Export analysis results to various formats."""
    
    def __init__(self):
        """Initialize the format exporter."""
        pass
    
    def export(
        self,
        result: AnalysisResult,
        export_format: ExportFormat,
        output_path: Union[str, Path],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Export analysis result to specified format.
        
        Args:
            result: Analysis result to export
            export_format: Target export format
            output_path: Output file path
            additional_data: Additional analysis data to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = result.to_dict()
        if additional_data:
            data.update(additional_data)
        
        # Export based on format
        if export_format == ExportFormat.JSON:
            self._export_json(data, output_path)
        elif export_format == ExportFormat.CSV:
            self._export_csv(result, output_path)
        elif export_format == ExportFormat.TSV:
            self._export_tsv(result, output_path)
        elif export_format == ExportFormat.XML:
            self._export_xml(data, output_path)
        elif export_format == ExportFormat.YAML:
            self._export_yaml(data, output_path)
        elif export_format == ExportFormat.ASE:
            self._export_ase(result.color_data.colors, output_path)
        elif export_format == ExportFormat.ACO:
            self._export_aco(result.color_data.colors, output_path)
        elif export_format == ExportFormat.GPL:
            self._export_gpl(result.color_data.colors, output_path)
        elif export_format == ExportFormat.SCSS:
            self._export_scss(result.color_data.colors, output_path)
        elif export_format == ExportFormat.CSS:
            self._export_css(result.color_data.colors, output_path)
        elif export_format == ExportFormat.LESS:
            self._export_less(result.color_data.colors, output_path)
        elif export_format == ExportFormat.STYLUS:
            self._export_stylus(result.color_data.colors, output_path)
        elif export_format == ExportFormat.SWIFT:
            self._export_swift(result.color_data.colors, output_path)
        elif export_format == ExportFormat.KOTLIN:
            self._export_kotlin(result.color_data.colors, output_path)
        elif export_format == ExportFormat.JAVA:
            self._export_java(result.color_data.colors, output_path)
        elif export_format == ExportFormat.PYTHON:
            self._export_python(result.color_data.colors, output_path)
        elif export_format == ExportFormat.JAVASCRIPT:
            self._export_javascript(result.color_data.colors, output_path)
        elif export_format == ExportFormat.HTML:
            self._export_html(result, output_path, additional_data)
        elif export_format == ExportFormat.SVG:
            self._export_svg(result.color_data.colors, output_path)
        elif export_format == ExportFormat.PNG:
            self._export_png_palette(result.color_data.colors, output_path)
        elif export_format == ExportFormat.PDF:
            self._export_pdf_report(result, output_path, additional_data)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export to JSON format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def _export_csv(self, result: AnalysisResult, output_path: Path) -> None:
        """Export colors to CSV format."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Hex', 'RGB_R', 'RGB_G', 'RGB_B', 'HSV_H', 'HSV_S', 'HSV_V',
                'Name', 'Frequency', 'Percentage', 'Luminance'
            ])
            
            # Data rows
            for color in result.color_data.colors:
                writer.writerow([
                    color.hex,
                    color.rgb[0], color.rgb[1], color.rgb[2],
                    f"{color.hsv[0]*360:.1f}" if color.hsv else "",
                    f"{color.hsv[1]*100:.1f}" if color.hsv else "",
                    f"{color.hsv[2]*100:.1f}" if color.hsv else "",
                    color.name or "",
                    color.frequency,
                    f"{color.percentage:.2f}",
                    f"{color.luminance:.4f}" if color.luminance else ""
                ])
    
    def _export_tsv(self, result: AnalysisResult, output_path: Path) -> None:
        """Export colors to TSV format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("Hex\tRGB_R\tRGB_G\tRGB_B\tHSV_H\tHSV_S\tHSV_V\tName\tFrequency\tPercentage\tLuminance\n")
            
            # Data rows
            for color in result.color_data.colors:
                f.write(f"{color.hex}\t{color.rgb[0]}\t{color.rgb[1]}\t{color.rgb[2]}\t")
                f.write(f"{color.hsv[0]*360:.1f}\t{color.hsv[1]*100:.1f}\t{color.hsv[2]*100:.1f}\t" if color.hsv else "\t\t\t")
                f.write(f"{color.name or ''}\t{color.frequency}\t{color.percentage:.2f}\t")
                f.write(f"{color.luminance:.4f}\n" if color.luminance else "\n")
    
    def _export_xml(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export to XML format."""
        root = ET.Element("colorAnalysis")
        
        def dict_to_xml(parent, data_dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    child = ET.SubElement(parent, key)
                    dict_to_xml(child, value)
                elif isinstance(value, list):
                    child = ET.SubElement(parent, key)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_elem = ET.SubElement(child, f"item_{i}")
                            dict_to_xml(item_elem, item)
                        else:
                            item_elem = ET.SubElement(child, f"item_{i}")
                            item_elem.text = str(item)
                else:
                    child = ET.SubElement(parent, key)
                    child.text = str(value)
        
        dict_to_xml(root, data)
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _export_yaml(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export to YAML format."""
        try:
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Fallback to simple YAML-like format
            with open(output_path, 'w', encoding='utf-8') as f:
                self._write_simple_yaml(f, data, 0)
    
    def _write_simple_yaml(self, f, data, indent_level):
        """Write simple YAML-like format."""
        indent = "  " * indent_level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    f.write(f"{indent}{key}:\n")
                    self._write_simple_yaml(f, value, indent_level + 1)
                else:
                    f.write(f"{indent}{key}: {value}\n")
        elif isinstance(data, list):
            for item in data:
                f.write(f"{indent}- ")
                if isinstance(item, (dict, list)):
                    f.write("\n")
                    self._write_simple_yaml(f, item, indent_level + 1)
                else:
                    f.write(f"{item}\n")
    
    def _export_ase(self, colors: list, output_path: Path) -> None:
        """Export to Adobe Swatch Exchange format."""
        # ASE format implementation
        with open(output_path, 'wb') as f:
            # ASE Header
            f.write(b'ASEF')  # File signature
            f.write(struct.pack('>HH', 1, 0))  # Version (1.0)
            
            # Number of blocks
            num_blocks = len(colors)
            f.write(struct.pack('>I', num_blocks))
            
            # Color blocks
            for i, color in enumerate(colors):
                # Block type (0x0001 = Color Entry)
                f.write(struct.pack('>H', 0x0001))
                
                # Block length (placeholder, will be updated)
                block_start = f.tell()
                f.write(struct.pack('>I', 0))
                
                # Color name
                name = color.name or f"Color {i+1}"
                name_utf16 = name.encode('utf-16be')
                f.write(struct.pack('>H', len(name_utf16) // 2 + 1))
                f.write(name_utf16)
                f.write(b'\x00\x00')  # Null terminator
                
                # Color model (RGB)
                f.write(b'RGB ')
                
                # RGB values (as floats)
                r, g, b = [c / 255.0 for c in color.rgb]
                f.write(struct.pack('>fff', r, g, b))
                
                # Color type (0 = Global, 1 = Spot, 2 = Normal)
                f.write(struct.pack('>H', 2))
                
                # Update block length
                block_end = f.tell()
                block_length = block_end - block_start - 4
                f.seek(block_start)
                f.write(struct.pack('>I', block_length))
                f.seek(block_end)
    
    def _export_aco(self, colors: list, output_path: Path) -> None:
        """Export to Adobe Color format."""
        with open(output_path, 'wb') as f:
            # ACO Header
            f.write(struct.pack('>HH', 1, len(colors)))  # Version 1, number of colors
            
            # Color entries
            for color in colors:
                f.write(struct.pack('>H', 0))  # Color space (0 = RGB)
                r, g, b = color.rgb
                f.write(struct.pack('>HHHH', 
                    r * 257, g * 257, b * 257, 0))  # RGB values (16-bit) + padding
    
    def _export_gpl(self, colors: list, output_path: Path) -> None:
        """Export to GIMP Palette format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("GIMP Palette\n")
            f.write("Name: PyColour Extract Palette\n")
            f.write("Columns: 8\n")
            f.write("#\n")
            
            for i, color in enumerate(colors):
                r, g, b = color.rgb
                name = color.name or f"Color_{i+1}"
                f.write(f"{r:3d} {g:3d} {b:3d}\t{name}\n")
    
    def _export_scss(self, colors: list, output_path: Path) -> None:
        """Export to SCSS variables format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - SCSS Color Variables\n\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f"${var_name}: {color.hex};\n")
            
            f.write("\n// Color palette array\n")
            f.write("$color-palette: (\n")
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f"  '{var_name}': {color.hex}")
                f.write(",\n" if i < len(colors) - 1 else "\n")
            f.write(");\n")
    
    def _export_css(self, colors: list, output_path: Path) -> None:
        """Export to CSS custom properties format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("/* PyColour Extract - CSS Custom Properties */\n\n")
            f.write(":root {\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f"  --{var_name}: {color.hex};\n")
            
            f.write("}\n\n")
            
            # Utility classes
            f.write("/* Color utility classes */\n")
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f".bg-{var_name} {{ background-color: var(--{var_name}); }}\n")
                f.write(f".text-{var_name} {{ color: var(--{var_name}); }}\n")
                f.write(f".border-{var_name} {{ border-color: var(--{var_name}); }}\n\n")
    
    def _export_less(self, colors: list, output_path: Path) -> None:
        """Export to LESS variables format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - LESS Color Variables\n\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f"@{var_name}: {color.hex};\n")
    
    def _export_stylus(self, colors: list, output_path: Path) -> None:
        """Export to Stylus variables format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - Stylus Color Variables\n\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color-{i+1}"
                f.write(f"{var_name} = {color.hex}\n")
    
    def _export_swift(self, colors: list, output_path: Path) -> None:
        """Export to Swift color constants."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - Swift Color Constants\n")
            f.write("import UIKit\n\n")
            f.write("extension UIColor {\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"color{i+1}"
                r, g, b = [c / 255.0 for c in color.rgb]
                f.write(f"    static let {var_name} = UIColor(red: {r:.3f}, green: {g:.3f}, blue: {b:.3f}, alpha: 1.0)\n")
            
            f.write("}\n")
    
    def _export_kotlin(self, colors: list, output_path: Path) -> None:
        """Export to Kotlin color constants."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - Kotlin Color Constants\n")
            f.write("import android.graphics.Color\n\n")
            f.write("object Colors {\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"COLOR_{i+1}"
                var_name = var_name.upper()
                f.write(f"    const val {var_name} = Color.parseColor(\"{color.hex}\")\n")
            
            f.write("}\n")
    
    def _export_java(self, colors: list, output_path: Path) -> None:
        """Export to Java color constants."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - Java Color Constants\n")
            f.write("import java.awt.Color;\n\n")
            f.write("public class Colors {\n")
            
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"COLOR_{i+1}"
                var_name = var_name.upper()
                r, g, b = color.rgb
                f.write(f"    public static final Color {var_name} = new Color({r}, {g}, {b});\n")
            
            f.write("}\n")
    
    def _export_python(self, colors: list, output_path: Path) -> None:
        """Export to Python color constants."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('"""PyColour Extract - Python Color Constants"""\n\n')
            
            # RGB tuples
            f.write("# RGB color tuples\n")
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"COLOR_{i+1}"
                var_name = var_name.upper()
                f.write(f"{var_name} = {color.rgb}\n")
            
            f.write("\n# Hex color codes\n")
            for i, color in enumerate(colors):
                var_name = self._sanitize_name(color.name) if color.name else f"COLOR_{i+1}"
                var_name = var_name.upper() + "_HEX"
                f.write(f"{var_name} = \"{color.hex}\"\n")
            
            # Color palette dictionary
            f.write("\n# Color palette dictionary\n")
            f.write("COLOR_PALETTE = {\n")
            for i, color in enumerate(colors):
                key_name = self._sanitize_name(color.name) if color.name else f"color_{i+1}"
                f.write(f"    '{key_name}': {color.rgb},\n")
            f.write("}\n")
    
    def _export_javascript(self, colors: list, output_path: Path) -> None:
        """Export to JavaScript color constants."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("// PyColour Extract - JavaScript Color Constants\n\n")
            
            # Color objects
            f.write("const colors = {\n")
            for i, color in enumerate(colors):
                key_name = self._sanitize_name(color.name) if color.name else f"color{i+1}"
                f.write(f"  {key_name}: {{\n")
                f.write(f"    hex: '{color.hex}',\n")
                f.write(f"    rgb: [{color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]}],\n")
                if color.hsv:
                    f.write(f"    hsv: [{color.hsv[0]*360:.1f}, {color.hsv[1]*100:.1f}, {color.hsv[2]*100:.1f}],\n")
                f.write(f"    name: '{color.name or ''}'\n")
                f.write(f"  }},\n")
            f.write("};\n\n")
            
            # Export for different module systems
            f.write("// CommonJS export\n")
            f.write("if (typeof module !== 'undefined' && module.exports) {\n")
            f.write("  module.exports = colors;\n")
            f.write("}\n\n")
            
            f.write("// ES6 export\n")
            f.write("export default colors;\n")
    
    def _export_html(self, result: AnalysisResult, output_path: Path, additional_data: Optional[Dict] = None) -> None:
        """Export to HTML color report."""
        colors = result.color_data.colors
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyColour Extract - Color Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            margin-top: 0;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .color-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .color-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .color-swatch {{
            height: 100px;
            width: 100%;
        }}
        .color-info {{
            padding: 15px;
            background: white;
        }}
        .color-hex {{
            font-family: monospace;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
        }}
        .color-name {{
            color: #666;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .color-stats {{
            font-size: 12px;
            color: #888;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .section {{
            margin: 30px 0;
        }}
        .meta {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Color Analysis Report</h1>
            <p>Generated by PyColour Extract</p>
        </div>
        
        <div class="meta">
            <h3>Image Information</h3>
            <p><strong>File:</strong> {result.color_data.image_metadata.filename}</p>
            <p><strong>Size:</strong> {result.color_data.image_metadata.size[0]} × {result.color_data.image_metadata.size[1]} pixels</p>
            <p><strong>Colors Found:</strong> {result.color_data.unique_color_count}</p>
            <p><strong>Algorithm:</strong> {result.algorithm_used}</p>
            <p><strong>Processing Time:</strong> {result.processing_time:.2f}s</p>
        </div>
        
        <div class="section">
            <h2>Color Palette</h2>
            <div class="color-grid">
"""
        
        # Add color cards
        for i, color in enumerate(colors[:16]):  # Limit to 16 colors for HTML
            html_content += f"""
                <div class="color-card">
                    <div class="color-swatch" style="background-color: {color.hex};"></div>
                    <div class="color-info">
                        <div class="color-hex">{color.hex}</div>
                        <div class="color-name">{color.name or 'Unknown'}</div>
                        <div class="color-stats">
                            RGB: ({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})<br>
                            Frequency: {color.percentage:.1f}%
                        </div>
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Color Details</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Preview</th>
                        <th>Hex</th>
                        <th>RGB</th>
                        <th>Name</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add table rows
        for color in colors[:20]:  # Limit to 20 colors for table
            html_content += f"""
                    <tr>
                        <td><div style="width: 30px; height: 30px; background: {color.hex}; border: 1px solid #ddd; border-radius: 3px;"></div></td>
                        <td><code>{color.hex}</code></td>
                        <td>({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})</td>
                        <td>{color.name or 'Unknown'}</td>
                        <td>{color.percentage:.2f}%</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_svg(self, colors: list, output_path: Path) -> None:
        """Export to SVG palette format."""
        width = 800
        height = 100
        swatch_width = width // len(colors) if colors else width
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <title>PyColour Extract Palette</title>
'''
        
        for i, color in enumerate(colors):
            x = i * swatch_width
            svg_content += f'''  <rect x="{x}" y="0" width="{swatch_width}" height="{height}" fill="{color.hex}" stroke="#000" stroke-width="0.5">
    <title>{color.name or 'Unknown'} - {color.hex}</title>
  </rect>
'''
        
        svg_content += '</svg>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
    
    def _export_png_palette(self, colors: list, output_path: Path) -> None:
        """Export palette as PNG image."""
        if not colors:
            return
        
        # Create palette image
        swatch_size = 100
        cols = min(8, len(colors))
        rows = (len(colors) + cols - 1) // cols
        
        width = cols * swatch_size
        height = rows * swatch_size
        
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        for i, color in enumerate(colors):
            row = i // cols
            col = i % cols
            
            x1 = col * swatch_size
            y1 = row * swatch_size
            x2 = x1 + swatch_size
            y2 = y1 + swatch_size
            
            draw.rectangle([(x1, y1), (x2, y2)], fill=color.rgb, outline='black')
        
        image.save(output_path)
    
    def _export_pdf_report(self, result: AnalysisResult, output_path: Path, additional_data: Optional[Dict] = None) -> None:
        """Export comprehensive PDF report using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.patches as mpatches
            
            with PdfPages(output_path) as pdf:
                # Page 1: Color Palette
                fig, axes = plt.subplots(2, 4, figsize=(11, 8.5))
                fig.suptitle('PyColour Extract - Color Analysis Report', fontsize=16, fontweight='bold')
                
                colors = result.color_data.colors[:8]
                
                for i, (ax, color) in enumerate(zip(axes.flat, colors)):
                    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=f"{color.hex}"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    ax.set_title(f"{color.hex}\n{color.name or 'Unknown'}\n{color.percentage:.1f}%", 
                               fontsize=10, ha='center')
                
                # Hide unused subplots
                for i in range(len(colors), len(axes.flat)):
                    axes.flat[i].axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Page 2: Statistics (if space allows)
                if len(result.color_data.colors) > 8:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    
                    # Create color distribution chart
                    top_colors = result.color_data.colors[:10]
                    percentages = [c.percentage for c in top_colors]
                    color_values = [c.hex for c in top_colors]
                    labels = [c.name or f"Color {i+1}" for i, c in enumerate(top_colors)]
                    
                    wedges, texts, autotexts = ax.pie(percentages, labels=labels, colors=color_values, 
                                                     autopct='%1.1f%%', startangle=90)
                    ax.set_title('Color Distribution', fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        except ImportError:
            # Fallback: create a simple text report
            with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write("PyColour Extract - Color Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Image: {result.color_data.image_metadata.filename}\n")
                f.write(f"Size: {result.color_data.image_metadata.size[0]} × {result.color_data.image_metadata.size[1]}\n")
                f.write(f"Colors Found: {result.color_data.unique_color_count}\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n\n")
                
                f.write("Color Palette:\n")
                f.write("-" * 20 + "\n")
                
                for i, color in enumerate(result.color_data.colors):
                    f.write(f"{i+1:2d}. {color.hex} - {color.name or 'Unknown'} ({color.percentage:.2f}%)\n")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize color name for use as variable name."""
        if not name:
            return "unknown"
        
        # Convert to lowercase and replace spaces/special chars with underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
        
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        return sanitized or 'unknown'
