"""Command-line interface for PyColour Extract."""

import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree
import typer
from PIL import Image

from .core.color_extractor import ColorExtractor
from .core.color_analyzer import ColorAnalyzer
from .exporters.format_exporter import FormatExporter
from .models.color_data import ExportFormat, AnalysisResult, ColorInfo


app = typer.Typer(
    name="pycolour-extract",
    help="Advanced color extraction and analysis tool",
    no_args_is_help=True
)
console = Console()


def version_callback(value: bool):
    """Show version information."""
    if value:
        from . import __version__
        console.print(f"PyColour Extract v{__version__}")
        raise typer.Exit()


@app.command()
def extract(
    image_path: Path = typer.Argument(..., help="Path to image file", exists=True, file_okay=True, dir_okay=False),
    algorithm: str = typer.Option("kmeans", help="Extraction algorithm"),
    max_colors: int = typer.Option(16, help="Maximum number of colors to extract"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
    export_formats: List[str] = typer.Option(["json"], help="Export formats"),
    show_preview: bool = typer.Option(True, help="Show color preview"),
    analyze_harmony: bool = typer.Option(True, help="Analyze color harmonies"),
    analyze_accessibility: bool = typer.Option(True, help="Analyze accessibility"),
    analyze_emotion: bool = typer.Option(False, help="Analyze emotional associations"),
    analyze_brand: bool = typer.Option(False, help="Analyze brand personality"),
    cluster_colors: bool = typer.Option(True, help="Perform color clustering"),
    n_clusters: int = typer.Option(5, help="Number of clusters for analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet output"),
    version: bool = typer.Option(False, "--version", callback=version_callback, help="Show version")
):
    """
    Extract colors from an image with advanced analysis.
    
    [bold green]Examples:[/bold green]
    
    [dim]# Basic extraction[/dim]
    [cyan]pycolour-extract image.jpg[/cyan]
    
    [dim]# Advanced analysis with multiple formats[/dim]
    [cyan]pycolour-extract image.jpg --algorithm kmeans --export-formats json css scss[/cyan]
    
    [dim]# Full analysis with branding insights[/dim]
    [cyan]pycolour-extract image.jpg --analyze-emotion --analyze-brand[/cyan]
    """
    if quiet:
        console.quiet = True
    
    # Validate algorithm
    valid_algorithms = ["kmeans", "dbscan", "median_cut", "octree", "histogram"]
    if algorithm not in valid_algorithms:
        console.print(f"[red]Error:[/red] Invalid algorithm '{algorithm}'. Valid algorithms: {', '.join(valid_algorithms)}")
        raise typer.Exit(1)
    
    # Validate export formats
    valid_formats = [fmt.value for fmt in ExportFormat]
    for fmt in export_formats:
        if fmt not in valid_formats:
            console.print(f"[red]Error:[/red] Invalid export format '{fmt}'. Valid formats: {', '.join(valid_formats)}")
            raise typer.Exit(1)
    
    # Setup output directory
    if output_dir is None:
        output_dir = image_path.parent / f"{image_path.stem}_analysis"
    output_dir.mkdir(exist_ok=True)
    
    if not quiet:
        console.print(f"\n[bold blue]🎨 PyColour Extract - Advanced Color Analysis[/bold blue]")
        console.print(f"[dim]Processing: {image_path}[/dim]\n")
    
    try:
        # Initialize extractor and analyzer
        extractor = ColorExtractor(algorithm=algorithm, max_colors=max_colors)
        analyzer = ColorAnalyzer()
        
        # Extract colors with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet
        ) as progress:
            
            task = progress.add_task("Extracting colors...", total=None)
            
            # Perform extraction
            result = extractor.extract_colors(
                image_path,
                n_clusters=n_clusters if cluster_colors else 0
            )
            
            progress.update(task, description="Analyzing colors...")
            
            # Additional analysis
            additional_analysis = {}
            
            if analyze_emotion:
                progress.update(task, description="Analyzing emotional associations...")
                additional_analysis['emotions'] = analyzer.calculate_color_emotion(result.color_data.colors)
            
            if analyze_brand:
                progress.update(task, description="Analyzing brand personality...")
                additional_analysis['brand_personality'] = analyzer.analyze_brand_personality(result.color_data.colors)
            
            if analyze_accessibility:
                progress.update(task, description="Analyzing color blindness impact...")
                additional_analysis['color_blindness'] = analyzer.calculate_color_blindness_impact(result.color_data.colors)
            
            progress.update(task, description="Generating analysis...")
            relationships = analyzer.analyze_color_relationships(result.color_data.colors)
            additional_analysis['relationships'] = relationships
        
        # Display results
        if not quiet:
            _display_results(result, additional_analysis, show_preview)
        
        # Export results
        exporter = FormatExporter()
        export_paths = []
        
        for fmt in export_formats:
            export_format = ExportFormat(fmt)
            output_path = output_dir / f"colors.{fmt}"
            
            exporter.export(result, export_format, output_path, additional_data=additional_analysis)
            export_paths.append(output_path)
        
        if not quiet:
            console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
            console.print(f"[dim]Processing time: {result.processing_time:.2f}s[/dim]")
            console.print(f"[dim]Output directory: {output_dir}[/dim]")
            
            if export_paths:
                console.print("\n[bold]Exported files:[/bold]")
                for path in export_paths:
                    console.print(f"  • {path.name}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing images", exists=True, file_okay=False, dir_okay=True),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
    algorithm: str = typer.Option("kmeans", help="Extraction algorithm"),
    max_colors: int = typer.Option(16, help="Maximum number of colors to extract"),
    export_formats: List[str] = typer.Option(["json"], help="Export formats"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process subdirectories recursively"),
    extensions: List[str] = typer.Option(["jpg", "jpeg", "png", "bmp", "tiff"], help="File extensions to process"),
    parallel: bool = typer.Option(True, help="Process images in parallel"),
    max_workers: int = typer.Option(4, help="Maximum number of parallel workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Batch process multiple images in a directory.
    
    [bold green]Examples:[/bold green]
    
    [dim]# Process all images in directory[/dim]
    [cyan]pycolour-extract batch ./images[/cyan]
    
    [dim]# Recursive processing with custom output[/dim]
    [cyan]pycolour-extract batch ./photos --recursive --output-dir ./analysis[/cyan]
    """
    if output_dir is None:
        output_dir = input_dir / "color_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Find image files
    patterns = [f"*.{ext}" for ext in extensions]
    image_files = []
    
    for pattern in patterns:
        if recursive:
            image_files.extend(input_dir.rglob(pattern))
        else:
            image_files.extend(input_dir.glob(pattern))
    
    if not image_files:
        console.print(f"[yellow]No image files found in {input_dir}[/yellow]")
        return
    
    console.print(f"\n[bold blue]📁 Batch Processing[/bold blue]")
    console.print(f"[dim]Found {len(image_files)} image files[/dim]")
    console.print(f"[dim]Algorithm: {algorithm}[/dim]")
    console.print(f"[dim]Max colors: {max_colors}[/dim]\n")
    
    extractor = ColorExtractor(algorithm=algorithm, max_colors=max_colors)
    exporter = FormatExporter()
    
    processed = 0
    errors = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("Processing images...", total=len(image_files))
        
        for image_file in image_files:
            try:
                # Create output subdirectory for this image
                rel_path = image_file.relative_to(input_dir)
                image_output_dir = output_dir / rel_path.parent / f"{rel_path.stem}_analysis"
                image_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract colors
                result = extractor.extract_colors(image_file)
                
                # Export in requested formats
                for fmt in export_formats:
                    export_format = ExportFormat(fmt)
                    output_path = image_output_dir / f"colors.{fmt}"
                    exporter.export(result, export_format, output_path)
                
                processed += 1
                
            except Exception as e:
                errors.append((image_file, str(e)))
                if verbose:
                    console.print(f"[red]Error processing {image_file}: {e}[/red]")
            
            progress.update(task, advance=1)
    
    console.print(f"\n[bold green]✓ Batch processing complete![/bold green]")
    console.print(f"[dim]Processed: {processed}/{len(image_files)} images[/dim]")
    
    if errors:
        console.print(f"[yellow]Errors: {len(errors)}[/yellow]")
        if verbose:
            for image_file, error in errors:
                console.print(f"  • {image_file}: {error}")


@app.command()
def compare(
    image_paths: List[Path] = typer.Argument(..., help="Paths to image files"),
    algorithm: str = typer.Option("kmeans", help="Extraction algorithm"),
    max_colors: int = typer.Option(8, help="Maximum number of colors to extract"),
    output_file: Optional[Path] = typer.Option(None, help="Output comparison file"),
    show_similarity: bool = typer.Option(True, help="Show color similarity analysis"),
    show_preview: bool = typer.Option(True, help="Show color previews")
):
    """
    Compare color palettes between multiple images.
    
    [bold green]Examples:[/bold green]
    
    [dim]# Compare two images[/dim]
    [cyan]pycolour-extract compare image1.jpg image2.jpg[/cyan]
    
    [dim]# Compare with similarity analysis[/dim]
    [cyan]pycolour-extract compare *.jpg --show-similarity[/cyan]
    """
    if len(image_paths) < 2:
        console.print("[red]Error:[/red] Need at least 2 images to compare")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]⚖️ Color Palette Comparison[/bold blue]")
    console.print(f"[dim]Comparing {len(image_paths)} images[/dim]\n")
    
    extractor = ColorExtractor(algorithm=algorithm, max_colors=max_colors)
    analyzer = ColorAnalyzer()
    
    results = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("Analyzing images...", total=len(image_paths))
        
        for image_path in image_paths:
            try:
                result = extractor.extract_colors(image_path)
                results.append((image_path, result))
            except Exception as e:
                console.print(f"[red]Error processing {image_path}: {e}[/red]")
                continue
            
            progress.update(task, advance=1)
    
    if len(results) < 2:
        console.print("[red]Error:[/red] Could not process enough images for comparison")
        raise typer.Exit(1)
    
    # Display comparison
    _display_comparison(results, show_similarity, show_preview)
    
    # Export comparison if requested
    if output_file:
        comparison_data = {
            "comparison": {
                "images": [],
                "similarity_matrix": [],
                "common_colors": []
            }
        }
        
        for image_path, result in results:
            comparison_data["comparison"]["images"].append({
                "path": str(image_path),
                "colors": [color.to_dict() for color in result.color_data.colors],
                "dominant_color": result.color_data.dominant_color.to_dict()
            })
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        console.print(f"\n[green]Comparison saved to: {output_file}[/green]")


@app.command()
def suggest(
    colors: List[str] = typer.Argument(..., help="Base colors (hex codes like #FF0000)"),
    suggestion_type: str = typer.Option("harmony", help="Type of suggestions"),
    count: int = typer.Option(5, help="Number of suggestions to generate"),
    export_format: str = typer.Option("json", help="Export format for suggestions")
):
    """
    Generate color suggestions based on input colors.
    
    [bold green]Examples:[/bold green]
    
    [dim]# Generate harmony suggestions[/dim]
    [cyan]pycolour-extract suggest "#FF0000" "#00FF00"[/cyan]
    
    [dim]# Generate accessible color pairs[/dim]
    [cyan]pycolour-extract suggest "#3366CC" --suggestion-type accessibility[/cyan]
    """
    # Parse input colors
    color_infos = []
    
    for color_hex in colors:
        try:
            if not color_hex.startswith('#'):
                color_hex = '#' + color_hex
            
            # Convert hex to RGB
            rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            
            color_info = ColorInfo(
                rgb=rgb,
                hex=color_hex,
                frequency=1,
                percentage=100.0 / len(colors)
            )
            color_infos.append(color_info)
            
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid color format: {color_hex}")
            raise typer.Exit(1)
    
    # Validate suggestion type
    valid_suggestion_types = ["harmony", "monochromatic", "analogous", "accessibility"]
    if suggestion_type not in valid_suggestion_types:
        console.print(f"[red]Error:[/red] Invalid suggestion type '{suggestion_type}'. Valid types: {', '.join(valid_suggestion_types)}")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]🎨 Color Suggestions[/bold blue]")
    console.print(f"[dim]Base colors: {', '.join(colors)}[/dim]")
    console.print(f"[dim]Suggestion type: {suggestion_type}[/dim]\n")
    
    # Generate suggestions
    analyzer = ColorAnalyzer()
    suggestions = analyzer.generate_color_suggestions(color_infos, suggestion_type)[:count]
    
    if not suggestions:
        console.print("[yellow]No suggestions could be generated for the given colors[/yellow]")
        return
    
    # Display suggestions
    _display_color_grid("Suggested Colors", suggestions)
    
    # Export if requested
    if export_format:
        suggestions_data = {
            "base_colors": [color.to_dict() for color in color_infos],
            "suggestion_type": suggestion_type,
            "suggestions": [color.to_dict() for color in suggestions]
        }
        
        output_file = f"color_suggestions.{export_format}"
        
        if export_format == "json":
            with open(output_file, 'w') as f:
                json.dump(suggestions_data, f, indent=2)
        
        console.print(f"\n[green]Suggestions saved to: {output_file}[/green]")


def _display_results(result: AnalysisResult, additional_analysis: Dict[str, Any], show_preview: bool = True):
    """Display extraction results in a formatted way."""
    color_data = result.color_data
    
    # Image info panel
    img_info = Table.grid()
    img_info.add_row("[bold]Image:[/bold]", color_data.image_metadata.filename)
    img_info.add_row("[bold]Size:[/bold]", f"{color_data.image_metadata.size[0]}×{color_data.image_metadata.size[1]} pixels")
    img_info.add_row("[bold]Format:[/bold]", color_data.image_metadata.format)
    img_info.add_row("[bold]Colors Found:[/bold]", str(color_data.unique_color_count))
    img_info.add_row("[bold]Algorithm:[/bold]", result.algorithm_used)
    
    if color_data.color_temperature:
        img_info.add_row("[bold]Color Temp:[/bold]", f"{color_data.color_temperature:.0f}K")
    if color_data.vibrance:
        img_info.add_row("[bold]Vibrance:[/bold]", f"{color_data.vibrance:.2f}")
    if color_data.accessibility_score:
        img_info.add_row("[bold]Accessibility:[/bold]", f"{color_data.accessibility_score:.2f}/1.0")
    
    console.print(Panel(img_info, title="[bold blue]Image Analysis[/bold blue]", border_style="blue"))
    
    # Color preview
    if show_preview:
        _display_color_grid("Dominant Colors", color_data.colors[:8])
    
    # Detailed color table
    _display_color_table(color_data.colors[:10])
    
    # Harmonies
    if color_data.harmonies:
        console.print("\n[bold cyan]🎵 Color Harmonies[/bold cyan]")
        for harmony in color_data.harmonies[:3]:
            console.print(f"  • [bold]{harmony.harmony_type.title()}[/bold] (confidence: {harmony.confidence:.2f})")
    
    # Clusters
    if color_data.clusters:
        console.print(f"\n[bold cyan]🔍 Color Clusters[/bold cyan]")
        for cluster in color_data.clusters[:3]:
            console.print(f"  • Cluster {cluster.cluster_id}: {cluster.size} colors, variance: {cluster.variance:.1f}")
    
    # Emotional analysis
    if 'emotions' in additional_analysis:
        emotions = additional_analysis['emotions']
        console.print(f"\n[bold magenta]💭 Emotional Associations[/bold magenta]")
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, score in top_emotions:
            console.print(f"  • [bold]{emotion.title()}:[/bold] {score:.2f}")
    
    # Brand personality
    if 'brand_personality' in additional_analysis:
        brand = additional_analysis['brand_personality']
        console.print(f"\n[bold yellow]🏢 Brand Personality[/bold yellow]")
        for trait_info in brand.get('dominant_traits', [])[:3]:
            console.print(f"  • [bold]{trait_info['trait'].title()}:[/bold] {trait_info['score']:.2f}")


def _display_color_grid(title: str, colors: List[ColorInfo], max_colors: int = 8):
    """Display colors in a visual grid."""
    if not colors:
        return
    
    color_panels = []
    for color in colors[:max_colors]:
        # Create color preview using colored text
        color_text = Text("████", style=f"bold {color.hex}")
        color_panel = Panel(
            f"{color_text}\n{color.hex}\n{color.percentage:.1f}%",
            title=color.name or "Unknown",
            border_style=color.hex,
            width=12,
            height=6
        )
        color_panels.append(color_panel)
    
    # Display in columns
    columns = Columns(color_panels, equal=True, expand=False)
    console.print(Panel(columns, title=f"[bold]{title}[/bold]"))


def _display_color_table(colors: List[ColorInfo]):
    """Display detailed color information in a table."""
    table = Table(title="Color Details")
    
    table.add_column("Preview", style="bold")
    table.add_column("Hex", style="cyan")
    table.add_column("RGB", style="green")
    table.add_column("HSV", style="yellow")
    table.add_column("Name", style="magenta")
    table.add_column("Frequency", justify="right")
    table.add_column("Percentage", justify="right")
    
    for color in colors:
        # Color preview using colored text
        preview = Text("██", style=f"bold {color.hex}")
        
        rgb_str = f"({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})"
        hsv_str = f"({color.hsv[0]*360:.0f}°, {color.hsv[1]*100:.0f}%, {color.hsv[2]*100:.0f}%)" if color.hsv else "N/A"
        
        table.add_row(
            preview,
            color.hex,
            rgb_str,
            hsv_str,
            color.name or "Unknown",
            f"{color.frequency:,}",
            f"{color.percentage:.2f}%"
        )
    
    console.print(table)


def _display_comparison(results: List[tuple], show_similarity: bool, show_preview: bool):
    """Display comparison results."""
    console.print("[bold cyan]📊 Palette Comparison[/bold cyan]\n")
    
    # Show individual palettes
    for i, (image_path, result) in enumerate(results):
        console.print(f"[bold]Image {i+1}: {image_path.name}[/bold]")
        if show_preview:
            _display_color_grid(f"Palette {i+1}", result.color_data.colors[:6], max_colors=6)
        console.print()
    
    # Show similarity analysis
    if show_similarity and len(results) >= 2:
        console.print("[bold cyan]🔍 Similarity Analysis[/bold cyan]")
        
        # Simple similarity based on dominant colors
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                image1_path, result1 = results[i]
                image2_path, result2 = results[j]
                
                # Calculate simple color distance between dominant colors
                dom1 = result1.color_data.dominant_color.rgb
                dom2 = result2.color_data.dominant_color.rgb
                distance = sum((a - b) ** 2 for a, b in zip(dom1, dom2)) ** 0.5
                similarity = max(0, 1 - distance / (255 * 3 ** 0.5))
                
                console.print(f"  • {image1_path.name} ↔ {image2_path.name}: {similarity:.2%} similar")


def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
