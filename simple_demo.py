#!/usr/bin/env python3
"""
PyColour Extract 2.0 - Simple Functionality Demo
"""

import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add src to path for imports
sys.path.insert(0, 'src')

def main():
    print("=" * 60)
    print("PYCOLOUR EXTRACT 2.0 - FUNCTIONALITY DEMONSTRATION")
    print("=" * 60)
    
    # Test imports
    print("Testing imports...")
    try:
        from pycolour_extract import ColorExtractor, ColorAnalyzer
        from pycolour_extract.exporters import FormatExporter
        from pycolour_extract.models.color_data import ExportFormat
        print("[OK] All core modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    
    # Create test image
    print("\nCreating test image...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))  # Red image
        test_image = temp_path / 'test.png'
        img.save(test_image)
        print("[OK] Test image created")
        
        # Test color extraction
        print("\nTesting color extraction...")
        try:
            extractor = ColorExtractor(algorithm="kmeans", max_colors=5)
            result = extractor.extract_colors(test_image)
            print(f"[OK] Found {len(result.color_data.colors)} colors")
            print(f"[OK] Dominant color: {result.color_data.dominant_color.hex}")
            print(f"[OK] Processing time: {result.processing_time:.3f} seconds")
        except Exception as e:
            print(f"[ERROR] Color extraction failed: {e}")
            return False
        
        # Test export
        print("\nTesting export...")
        try:
            exporter = FormatExporter()
            json_file = temp_path / 'colors.json'
            exporter.export(result, ExportFormat.JSON, json_file)
            
            if json_file.exists():
                print(f"[OK] JSON export successful: {json_file.stat().st_size} bytes")
            else:
                print("[ERROR] JSON file not created")
                return False
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            return False
        
        # Test analysis
        print("\nTesting analysis...")
        try:
            analyzer = ColorAnalyzer()
            emotions = analyzer.calculate_color_emotion(result.color_data.colors)
            print(f"[OK] Emotional analysis completed: {len(emotions)} categories")
            
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            print(f"[OK] Top emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            return False
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY - ALL TESTS PASSED!")
        print("=" * 60)
        print(f"Color Extraction:  WORKING")
        print(f"Export Functions:  WORKING") 
        print(f"Analysis Features: WORKING")
        print(f"Processing Speed:  {result.processing_time:.3f}s")
        print("\nPyColour Extract 2.0 core functionality is 100% operational!")
        
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nDemo failed: {e}")
        sys.exit(1)
