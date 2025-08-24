"""Integration tests for the complete workflow."""

import pytest
import json
import tempfile
from pathlib import Path
from PIL import Image

from pycolour_extract import ColorExtractor, ColorAnalyzer
from pycolour_extract.core.palette_generator import PaletteGenerator
from pycolour_extract.exporters.format_exporter import FormatExporter
from pycolour_extract.models.color_data import ExportFormat


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_analysis_workflow(self, sample_image, temp_dir):
        """Test complete analysis workflow from extraction to export."""
        # Step 1: Extract colors
        extractor = ColorExtractor(algorithm="kmeans", max_colors=8)
        result = extractor.extract_colors(sample_image)
        
        assert result is not None
        assert len(result.color_data.colors) > 0
        
        # Step 2: Analyze colors
        analyzer = ColorAnalyzer()
        emotions = analyzer.calculate_color_emotion(result.color_data.colors)
        brand_personality = analyzer.analyze_brand_personality(result.color_data.colors)
        relationships = analyzer.analyze_color_relationships(result.color_data.colors)
        
        assert emotions is not None
        assert brand_personality is not None
        assert relationships is not None
        
        # Step 3: Generate additional palettes
        generator = PaletteGenerator()
        additional_palettes = generator.generate_palette_variations(
            result.color_data.colors, "harmony", 5
        )
        
        # Step 4: Export results
        exporter = FormatExporter()
        
        # Export to multiple formats
        formats_to_test = [ExportFormat.JSON, ExportFormat.CSS, ExportFormat.HTML]
        
        for export_format in formats_to_test:
            output_path = temp_dir / f"complete_analysis.{export_format.value}"
            
            additional_data = {
                "emotions": emotions,
                "brand_personality": brand_personality,
                "relationships": relationships,
                "additional_palettes": [p.to_dict() for p in additional_palettes]
            }
            
            exporter.export(result, export_format, output_path, additional_data)
            assert output_path.exists()
            
            # Verify JSON content
            if export_format == ExportFormat.JSON:
                with open(output_path) as f:
                    data = json.load(f)
                
                assert "color_data" in data
                assert "emotions" in data
                assert "brand_personality" in data
                assert "relationships" in data
    
    def test_batch_processing_workflow(self, multiple_images, temp_dir):
        """Test batch processing workflow."""
        if not multiple_images:
            pytest.skip("No multiple images available for batch testing")
        
        extractor = ColorExtractor(algorithm="kmeans", max_colors=6)
        analyzer = ColorAnalyzer()
        exporter = FormatExporter()
        
        batch_results = []
        
        # Process each image
        for image_path in multiple_images:
            # Extract colors
            result = extractor.extract_colors(image_path)
            
            # Analyze
            emotions = analyzer.calculate_color_emotion(result.color_data.colors)
            
            # Store result
            batch_results.append({
                "image_path": str(image_path),
                "result": result,
                "emotions": emotions
            })
        
        # Export batch summary
        summary_data = {
            "batch_info": {
                "total_images": len(batch_results),
                "algorithm_used": "kmeans",
                "max_colors": 6
            },
            "results": []
        }
        
        for batch_result in batch_results:
            result_data = batch_result["result"].to_dict()
            result_data["emotions"] = batch_result["emotions"]
            summary_data["results"].append(result_data)
        
        summary_path = temp_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        assert summary_path.exists()
        
        # Verify summary content
        with open(summary_path) as f:
            data = json.load(f)
        
        assert data["batch_info"]["total_images"] == len(multiple_images)
        assert len(data["results"]) == len(multiple_images)
    
    def test_algorithm_comparison_workflow(self, sample_image, temp_dir):
        """Test comparing results from different algorithms."""
        algorithms = ["kmeans", "median_cut", "octree"]
        results = {}
        
        # Extract with different algorithms
        for algorithm in algorithms:
            extractor = ColorExtractor(algorithm=algorithm, max_colors=6)
            result = extractor.extract_colors(sample_image)
            results[algorithm] = result
        
        # Compare results
        comparison_data = {
            "image": str(sample_image),
            "algorithms": {}
        }
        
        for algorithm, result in results.items():
            comparison_data["algorithms"][algorithm] = {
                "unique_colors": result.color_data.unique_color_count,
                "processing_time": result.processing_time,
                "dominant_color": result.color_data.dominant_color.to_dict(),
                "colors": [c.to_dict() for c in result.color_data.colors]
            }
        
        # Export comparison
        comparison_path = temp_dir / "algorithm_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        assert comparison_path.exists()
        
        # Verify all algorithms produced results
        with open(comparison_path) as f:
            data = json.load(f)
        
        for algorithm in algorithms:
            assert algorithm in data["algorithms"]
            assert data["algorithms"][algorithm]["unique_colors"] > 0
            assert data["algorithms"][algorithm]["processing_time"] > 0
    
    def test_palette_generation_workflow(self, sample_image, temp_dir):
        """Test comprehensive palette generation workflow."""
        # Extract colors
        extractor = ColorExtractor(algorithm="kmeans", max_colors=10)
        result = extractor.extract_colors(sample_image)
        
        # Generate multiple palette styles
        generator = PaletteGenerator()
        
        palette_styles = ["dominant", "harmony", "monochromatic", "vibrant", "muted"]
        all_palettes = {}
        
        for style in palette_styles:
            palettes = generator.generate_palette_variations(
                result.color_data.colors, style, 5
            )
            all_palettes[style] = palettes
        
        # Create visual palettes
        visual_palettes = {}
        for style, palettes in all_palettes.items():
            if palettes:
                palette = palettes[0]  # Use first palette of each style
                image = generator.create_palette_image(
                    palette.colors, "swatches", 400, 100, True
                )
                
                # Save palette image
                image_path = temp_dir / f"palette_{style}.png"
                image.save(image_path)
                visual_palettes[style] = str(image_path)
        
        # Analyze palette moods
        mood_analyses = {}
        for style, palettes in all_palettes.items():
            if palettes:
                mood = generator.analyze_palette_mood(palettes[0].colors)
                mood_analyses[style] = mood
        
        # Export comprehensive palette data
        palette_data = {
            "source_image": str(sample_image),
            "base_colors": [c.to_dict() for c in result.color_data.colors],
            "palette_styles": {},
            "visual_palettes": visual_palettes,
            "mood_analyses": mood_analyses
        }
        
        for style, palettes in all_palettes.items():
            palette_data["palette_styles"][style] = [p.to_dict() for p in palettes]
        
        palette_export_path = temp_dir / "comprehensive_palettes.json"
        with open(palette_export_path, 'w') as f:
            json.dump(palette_data, f, indent=2, default=str)
        
        assert palette_export_path.exists()
        
        # Verify palette images were created
        for style in visual_palettes:
            image_path = Path(visual_palettes[style])
            assert image_path.exists()
            
            # Verify it's a valid image
            image = Image.open(image_path)
            assert image.size == (400, 100)
    
    def test_accessibility_workflow(self, sample_image, temp_dir):
        """Test comprehensive accessibility analysis workflow."""
        # Extract colors
        extractor = ColorExtractor(algorithm="kmeans", max_colors=8)
        result = extractor.extract_colors(sample_image)
        
        # Perform accessibility analysis
        analyzer = ColorAnalyzer()
        
        # Color blindness analysis
        cb_impact = analyzer.calculate_color_blindness_impact(result.color_data.colors)
        
        # Generate accessible color suggestions
        accessible_suggestions = analyzer.generate_color_suggestions(
            result.color_data.colors, "accessibility"
        )
        
        # Analyze color relationships for accessibility
        relationships = analyzer.analyze_color_relationships(result.color_data.colors)
        accessible_pairs = relationships.get("accessibility_pairs", [])
        
        # Compile accessibility report
        accessibility_report = {
            "image": str(sample_image),
            "color_count": len(result.color_data.colors),
            "accessibility_score": result.color_data.accessibility_score,
            "color_blindness_impact": cb_impact,
            "accessible_color_pairs": [
                {
                    "color1": pair[0].to_dict(),
                    "color2": pair[1].to_dict(),
                    "contrast_ratio": pair[2]
                } for pair in accessible_pairs[:5]  # Top 5 accessible pairs
            ],
            "suggested_improvements": [c.to_dict() for c in accessible_suggestions],
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if result.color_data.accessibility_score and result.color_data.accessibility_score < 0.5:
            accessibility_report["recommendations"].append(
                "Consider increasing contrast between colors"
            )
        
        if cb_impact["overall_accessibility"] < 0.4:
            accessibility_report["recommendations"].extend(
                cb_impact.get("recommendations", [])
            )
        
        # Export accessibility report
        accessibility_path = temp_dir / "accessibility_report.json"
        with open(accessibility_path, 'w') as f:
            json.dump(accessibility_report, f, indent=2, default=str)
        
        assert accessibility_path.exists()
        
        # Verify report content
        with open(accessibility_path) as f:
            data = json.load(f)
        
        assert "color_blindness_impact" in data
        assert "simulations" in data["color_blindness_impact"]
        assert "distinguishability_scores" in data["color_blindness_impact"]
    
    def test_brand_analysis_workflow(self, sample_image, temp_dir):
        """Test comprehensive brand analysis workflow."""
        # Extract colors
        extractor = ColorExtractor(algorithm="kmeans", max_colors=6)
        result = extractor.extract_colors(sample_image)
        
        # Perform brand analysis
        analyzer = ColorAnalyzer()
        
        # Emotional analysis
        emotions = analyzer.calculate_color_emotion(result.color_data.colors)
        
        # Brand personality
        brand_personality = analyzer.analyze_brand_personality(result.color_data.colors)
        
        # Color relationships
        relationships = analyzer.analyze_color_relationships(result.color_data.colors)
        
        # Palette mood analysis
        generator = PaletteGenerator()
        mood_analysis = generator.analyze_palette_mood(result.color_data.colors)
        
        # Palette improvement suggestions
        improvement_suggestions = generator.suggest_palette_improvements(result.color_data.colors)
        
        # Compile brand analysis report
        brand_report = {
            "image": str(sample_image),
            "extracted_colors": [c.to_dict() for c in result.color_data.colors],
            "color_temperature": result.color_data.color_temperature,
            "vibrance": result.color_data.vibrance,
            "emotional_analysis": emotions,
            "brand_personality": brand_personality,
            "mood_analysis": mood_analysis,
            "color_relationships": {
                "harmony_types": relationships.get("harmony_types", []),
                "temperature_analysis": relationships.get("temperature_analysis", {}),
                "complementary_pairs": len(relationships.get("complementary_pairs", [])),
                "analogous_groups": len(relationships.get("analogous_groups", []))
            },
            "improvement_suggestions": improvement_suggestions,
            "brand_recommendations": []
        }
        
        # Generate brand recommendations based on analysis
        dominant_traits = brand_personality.get("dominant_traits", [])
        if dominant_traits:
            top_trait = dominant_traits[0]["trait"]
            brand_report["brand_recommendations"].append(
                f"Color palette suggests a {top_trait} brand personality"
            )
        
        if emotions.get("trustworthiness", 0) > 0.7:
            brand_report["brand_recommendations"].append(
                "High trustworthiness score - suitable for financial or healthcare brands"
            )
        
        if emotions.get("energy", 0) > 0.7:
            brand_report["brand_recommendations"].append(
                "High energy score - suitable for sports or youth-oriented brands"
            )
        
        # Export brand report
        brand_path = temp_dir / "brand_analysis_report.json"
        with open(brand_path, 'w') as f:
            json.dump(brand_report, f, indent=2, default=str)
        
        assert brand_path.exists()
        
        # Verify report content
        with open(brand_path) as f:
            data = json.load(f)
        
        assert "emotional_analysis" in data
        assert "brand_personality" in data
        assert "mood_analysis" in data
        assert len(data["extracted_colors"]) > 0
    
    def test_export_format_compatibility(self, sample_image, temp_dir):
        """Test that all export formats work together."""
        # Extract colors
        extractor = ColorExtractor(algorithm="kmeans", max_colors=5)
        result = extractor.extract_colors(sample_image)
        
        # Test all export formats
        exporter = FormatExporter()
        
        export_formats = [
            ExportFormat.JSON, ExportFormat.CSV, ExportFormat.CSS, 
            ExportFormat.SCSS, ExportFormat.HTML, ExportFormat.SVG,
            ExportFormat.PNG, ExportFormat.PYTHON, ExportFormat.JAVASCRIPT
        ]
        
        exported_files = []
        
        for export_format in export_formats:
            output_path = temp_dir / f"export_test.{export_format.value}"
            
            try:
                exporter.export(result, export_format, output_path)
                if output_path.exists():
                    exported_files.append((export_format, output_path))
            except Exception as e:
                pytest.fail(f"Failed to export {export_format.value}: {e}")
        
        # Verify files were created
        assert len(exported_files) == len(export_formats)
        
        # Verify basic file content for text formats
        text_formats = [ExportFormat.JSON, ExportFormat.CSS, ExportFormat.PYTHON]
        
        for format_type, file_path in exported_files:
            if format_type in text_formats:
                content = file_path.read_text(encoding='utf-8')
                assert len(content) > 0
                
                # Check for color hex codes in content
                colors_found = any(
                    color.hex.lower() in content.lower() 
                    for color in result.color_data.colors
                )
                assert colors_found, f"No colors found in {format_type.value} export"
    
    def test_error_recovery_workflow(self, temp_dir):
        """Test error recovery in various workflow scenarios."""
        extractor = ColorExtractor()
        analyzer = ColorAnalyzer()
        exporter = FormatExporter()
        
        # Test with corrupted image file
        corrupt_file = temp_dir / "corrupt_image.jpg"
        corrupt_file.write_text("This is not an image file")
        
        with pytest.raises((IOError, OSError)):
            extractor.extract_colors(corrupt_file)
        
        # Test with empty color list
        empty_emotions = analyzer.calculate_color_emotion([])
        assert all(score == 0.0 for score in empty_emotions.values())
        
        empty_brand = analyzer.analyze_brand_personality([])
        assert "error" in empty_brand
        
        # Test export with minimal data
        from pycolour_extract.models.color_data import ImageMetadata, ColorData, AnalysisResult
        
        minimal_metadata = ImageMetadata(
            path="test.jpg", filename="test.jpg", size=(100, 100),
            format="JPEG", mode="RGB", total_pixels=10000, file_size=5000
        )
        
        minimal_color = ColorInfo(rgb=(128, 128, 128), hex="#808080")
        
        minimal_data = ColorData(
            image_metadata=minimal_metadata,
            colors=[minimal_color],
            unique_color_count=1,
            dominant_color=minimal_color,
            average_color=minimal_color
        )
        
        minimal_result = AnalysisResult(
            color_data=minimal_data,
            palettes=[],
            processing_time=0.1,
            algorithm_used="test",
            settings={}
        )
        
        # Should not raise errors
        output_path = temp_dir / "minimal_export.json"
        exporter.export(minimal_result, ExportFormat.JSON, output_path)
        assert output_path.exists()
    
    def test_performance_workflow(self, complex_image, temp_dir):
        """Test performance with complex images."""
        import time
        
        start_time = time.time()
        
        # Extract with performance settings
        extractor = ColorExtractor(algorithm="octree", max_colors=16)  # Fastest algorithm
        result = extractor.extract_colors(complex_image)
        
        extraction_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert extraction_time < 30.0, f"Extraction took too long: {extraction_time:.2f}s"
        
        # Verify quality of results
        assert len(result.color_data.colors) > 5
        assert result.color_data.unique_color_count > 5
        
        # Quick analysis
        analyzer = ColorAnalyzer()
        emotions = analyzer.calculate_color_emotion(result.color_data.colors[:5])  # Limit for speed
        
        assert len(emotions) == 8  # Should have all emotion categories
        
        # Export to lightweight format
        exporter = FormatExporter()
        output_path = temp_dir / "performance_test.json"
        exporter.export(result, ExportFormat.JSON, output_path)
        
        total_time = time.time() - start_time
        assert total_time < 35.0, f"Total workflow took too long: {total_time:.2f}s"
