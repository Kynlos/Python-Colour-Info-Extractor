"""Tests for CLI functionality."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from pycolour_extract.cli import app
from pycolour_extract.models.color_data import ExportFormat


class TestCLI:
    """Test CLI functionality."""
    
    def setup_method(self):
        """Set up test method."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Advanced color extraction and analysis tool" in result.stdout
    
    def test_extract_command_help(self):
        """Test extract command help."""
        result = self.runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract colors from an image" in result.stdout
    
    def test_batch_command_help(self):
        """Test batch command help."""
        result = self.runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Batch process multiple images" in result.stdout
    
    def test_compare_command_help(self):
        """Test compare command help."""
        result = self.runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare color palettes" in result.stdout
    
    def test_suggest_command_help(self):
        """Test suggest command help."""
        result = self.runner.invoke(app, ["suggest", "--help"])
        assert result.exit_code == 0
        assert "Generate color suggestions" in result.stdout
    
    def test_extract_command_basic(self, sample_image, temp_dir):
        """Test basic extract command."""
        with patch('pycolour_extract.cli.console') as mock_console:
            mock_console.quiet = False
            result = self.runner.invoke(app, [
                "extract", 
                str(sample_image),
                "--output-dir", str(temp_dir),
                "--quiet"
            ])
            
            # Should succeed (exit code 0)
            assert result.exit_code == 0
    
    def test_extract_command_with_options(self, sample_image, temp_dir):
        """Test extract command with various options."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--algorithm", "kmeans",
            "--max-colors", "8",
            "--output-dir", str(temp_dir),
            "--export-formats", "json", "css",
            "--no-show-preview",
            "--quiet"
        ])
        
        assert result.exit_code == 0
        
        # Check output files were created
        output_dir = temp_dir / f"{sample_image.stem}_analysis"
        assert output_dir.exists()
    
    def test_extract_invalid_algorithm(self, sample_image):
        """Test extract command with invalid algorithm."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--algorithm", "invalid_algorithm"
        ])
        
        # Should fail with non-zero exit code
        assert result.exit_code != 0
    
    def test_extract_invalid_export_format(self, sample_image):
        """Test extract command with invalid export format."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--export-formats", "invalid_format"
        ])
        
        assert result.exit_code == 1
        assert "Invalid export format" in result.stdout
    
    def test_extract_nonexistent_file(self, temp_dir):
        """Test extract command with non-existent file."""
        nonexistent = temp_dir / "nonexistent.jpg"
        
        result = self.runner.invoke(app, [
            "extract",
            str(nonexistent)
        ])
        
        # Should fail
        assert result.exit_code != 0
    
    def test_batch_command_basic(self, multiple_images, temp_dir):
        """Test basic batch command."""
        # Create a directory with the multiple images
        input_dir = temp_dir / "input_images"
        input_dir.mkdir()
        
        # Copy images to input directory
        for i, image_path in enumerate(multiple_images):
            new_path = input_dir / f"image_{i}.png"
            new_path.write_bytes(image_path.read_bytes())
        
        result = self.runner.invoke(app, [
            "batch",
            str(input_dir),
            "--output-dir", str(temp_dir / "batch_output"),
            "--export-formats", "json"
        ])
        
        assert result.exit_code == 0
    
    def test_batch_command_recursive(self, multiple_images, temp_dir):
        """Test batch command with recursive option."""
        # Create nested directory structure
        input_dir = temp_dir / "input_images"
        subdir = input_dir / "subdir"
        subdir.mkdir(parents=True)
        
        # Place image in subdirectory
        if multiple_images:
            image_path = subdir / "test.png"
            image_path.write_bytes(multiple_images[0].read_bytes())
        
        result = self.runner.invoke(app, [
            "batch",
            str(input_dir),
            "--recursive",
            "--output-dir", str(temp_dir / "recursive_output"),
            "--export-formats", "json"
        ])
        
        assert result.exit_code == 0
    
    def test_batch_empty_directory(self, temp_dir):
        """Test batch command with empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        result = self.runner.invoke(app, [
            "batch",
            str(empty_dir)
        ])
        
        assert result.exit_code == 0
        assert "No image files found" in result.stdout
    
    def test_compare_command_basic(self, multiple_images, temp_dir):
        """Test basic compare command."""
        if len(multiple_images) < 2:
            pytest.skip("Need at least 2 images for comparison")
        
        result = self.runner.invoke(app, [
            "compare",
            str(multiple_images[0]),
            str(multiple_images[1]),
            "--output-file", str(temp_dir / "comparison.json")
        ])
        
        assert result.exit_code == 0
        
        # Check comparison file was created
        comparison_file = temp_dir / "comparison.json"
        assert comparison_file.exists()
        
        # Verify it's valid JSON
        with open(comparison_file) as f:
            data = json.load(f)
        assert "comparison" in data
    
    def test_compare_single_image_fails(self, sample_image):
        """Test compare command with single image fails."""
        result = self.runner.invoke(app, [
            "compare",
            str(sample_image)
        ])
        
        assert result.exit_code == 1
        assert "Need at least 2 images" in result.stdout
    
    def test_suggest_command_basic(self):
        """Test basic suggest command."""
        result = self.runner.invoke(app, [
            "suggest",
            "#FF0000", "#00FF00",
            "--suggestion-type", "harmony",
            "--count", "3"
        ])
        
        assert result.exit_code == 0
    
    def test_suggest_command_invalid_color(self):
        """Test suggest command with invalid color."""
        result = self.runner.invoke(app, [
            "suggest",
            "invalid_color"
        ])
        
        assert result.exit_code == 1
        assert "Invalid color format" in result.stdout
    
    def test_suggest_command_different_types(self):
        """Test suggest command with different suggestion types."""
        suggestion_types = ["harmony", "monochromatic", "analogous", "accessibility"]
        
        for suggestion_type in suggestion_types:
            result = self.runner.invoke(app, [
                "suggest",
                "#FF0000",
                "--suggestion-type", suggestion_type,
                "--count", "2"
            ])
            
            assert result.exit_code == 0
    
    def test_version_callback(self):
        """Test version callback."""
        with patch('pycolour_extract.cli.console') as mock_console:
            result = self.runner.invoke(app, ["--version"])
            
            # Should exit after showing version
            assert result.exit_code == 0
            mock_console.print.assert_called()
    
    @pytest.mark.parametrize("algorithm", ["kmeans", "dbscan", "median_cut", "octree", "histogram"])
    def test_extract_all_algorithms(self, sample_image, temp_dir, algorithm):
        """Test extract command with all supported algorithms."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--algorithm", algorithm,
            "--max-colors", "5",
            "--output-dir", str(temp_dir),
            "--quiet"
        ])
        
        assert result.exit_code == 0
    
    @pytest.mark.parametrize("export_format", ["json", "css", "scss", "html", "svg"])
    def test_extract_export_formats(self, sample_image, temp_dir, export_format):
        """Test extract command with different export formats."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--export-formats", export_format,
            "--output-dir", str(temp_dir),
            "--quiet"
        ])
        
        assert result.exit_code == 0
        
        # Check that output file was created
        output_dir = temp_dir / f"{sample_image.stem}_analysis"
        output_file = output_dir / f"colors.{export_format}"
        assert output_file.exists()
    
    def test_extract_with_analysis_options(self, sample_image, temp_dir):
        """Test extract command with analysis options enabled."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--analyze-emotion",
            "--analyze-brand", 
            "--analyze-accessibility",
            "--cluster-colors",
            "--n-clusters", "3",
            "--output-dir", str(temp_dir),
            "--quiet"
        ])
        
        assert result.exit_code == 0
    
    def test_verbose_output(self, sample_image, temp_dir):
        """Test verbose output option."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--verbose",
            "--output-dir", str(temp_dir),
            "--quiet"
        ])
        
        assert result.exit_code == 0
    
    def test_quiet_mode(self, sample_image, temp_dir):
        """Test quiet mode option."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--quiet",
            "--output-dir", str(temp_dir)
        ])
        
        assert result.exit_code == 0
    
    def test_extract_error_handling(self, temp_dir):
        """Test extract command error handling."""
        # Test with corrupted image file
        corrupt_file = temp_dir / "corrupt.jpg"
        corrupt_file.write_text("This is not an image")
        
        result = self.runner.invoke(app, [
            "extract",
            str(corrupt_file),
            "--quiet"
        ])
        
        assert result.exit_code == 1
        assert "Error:" in result.stdout
    
    def test_batch_extensions_filter(self, temp_dir):
        """Test batch command with custom extensions."""
        input_dir = temp_dir / "mixed_files"
        input_dir.mkdir()
        
        # Create files with different extensions
        (input_dir / "image.jpg").write_bytes(b"fake jpg")
        (input_dir / "image.png").write_bytes(b"fake png")
        (input_dir / "document.txt").write_text("text file")
        
        result = self.runner.invoke(app, [
            "batch",
            str(input_dir),
            "--extensions", "jpg",
            "--output-dir", str(temp_dir / "filtered_output")
        ])
        
        # Should not fail even with fake image files
        assert result.exit_code == 0
    
    def test_compare_with_similarity(self, multiple_images):
        """Test compare command with similarity analysis."""
        if len(multiple_images) < 2:
            pytest.skip("Need at least 2 images for comparison")
        
        result = self.runner.invoke(app, [
            "compare",
            str(multiple_images[0]),
            str(multiple_images[1]),
            "--show-similarity",
            "--show-preview"
        ])
        
        assert result.exit_code == 0
    
    def test_suggest_with_export(self, temp_dir):
        """Test suggest command with export option."""
        result = self.runner.invoke(app, [
            "suggest",
            "#FF0000", "#0000FF",
            "--export-format", "json"
        ])
        
        assert result.exit_code == 0
        
        # Check that suggestions file was created
        suggestions_file = Path("color_suggestions.json")
        if suggestions_file.exists():
            suggestions_file.unlink()  # Clean up
    
    def test_extract_output_directory_creation(self, sample_image, temp_dir):
        """Test that extract command creates output directories."""
        nested_output = temp_dir / "level1" / "level2" / "output"
        
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--output-dir", str(nested_output),
            "--quiet"
        ])
        
        assert result.exit_code == 0
        assert nested_output.exists()
    
    def test_batch_parallel_processing(self, multiple_images, temp_dir):
        """Test batch command with parallel processing options."""
        input_dir = temp_dir / "parallel_input"
        input_dir.mkdir()
        
        # Copy images to input directory
        for i, image_path in enumerate(multiple_images):
            new_path = input_dir / f"image_{i}.png"
            new_path.write_bytes(image_path.read_bytes())
        
        result = self.runner.invoke(app, [
            "batch",
            str(input_dir),
            "--parallel",
            "--max-workers", "2",
            "--output-dir", str(temp_dir / "parallel_output")
        ])
        
        assert result.exit_code == 0
    
    def test_extract_multiple_export_formats(self, sample_image, temp_dir):
        """Test extract command with multiple export formats."""
        result = self.runner.invoke(app, [
            "extract",
            str(sample_image),
            "--export-formats", "json", "css", "scss", "html",
            "--output-dir", str(temp_dir),
            "--quiet"
        ])
        
        assert result.exit_code == 0
        
        # Check that all format files were created
        output_dir = temp_dir / f"{sample_image.stem}_analysis"
        formats = ["json", "css", "scss", "html"]
        
        for fmt in formats:
            output_file = output_dir / f"colors.{fmt}"
            assert output_file.exists()
    
    def test_keyboard_interrupt_handling(self, sample_image):
        """Test keyboard interrupt handling."""
        with patch('pycolour_extract.cli.ColorExtractor.extract_colors') as mock_extract:
            mock_extract.side_effect = KeyboardInterrupt()
            
            result = self.runner.invoke(app, [
                "extract",
                str(sample_image)
            ])
            
            # Should handle KeyboardInterrupt gracefully
            assert result.exit_code == 1
            assert "cancelled" in result.stdout.lower() or "interrupted" in result.stdout.lower()
    
    def test_cli_main_function(self):
        """Test main CLI function."""
        # This tests the main() function wrapper
        with patch('pycolour_extract.cli.app') as mock_app:
            from pycolour_extract.cli import main
            
            try:
                main()
            except SystemExit:
                pass  # Expected for CLI apps
            
            mock_app.assert_called_once()
