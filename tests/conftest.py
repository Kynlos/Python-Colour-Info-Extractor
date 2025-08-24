"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from pycolour_extract.models.color_data import ColorInfo, ColorData, ImageMetadata


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image with known colors."""
    # Create a simple test image with distinct color blocks
    width, height = 200, 200
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    # Create quadrants with different colors
    # Red quadrant (top-left)
    draw.rectangle([(0, 0), (width//2, height//2)], fill=(255, 0, 0))
    # Green quadrant (top-right)
    draw.rectangle([(width//2, 0), (width, height//2)], fill=(0, 255, 0))
    # Blue quadrant (bottom-left)
    draw.rectangle([(0, height//2), (width//2, height)], fill=(0, 0, 255))
    # Yellow quadrant (bottom-right)
    draw.rectangle([(width//2, height//2), (width, height)], fill=(255, 255, 0))
    
    image_path = temp_dir / "test_image.png"
    image.save(image_path)
    return image_path


@pytest.fixture
def gradient_image(temp_dir):
    """Create a gradient test image."""
    width, height = 200, 100
    image = Image.new('RGB', (width, height))
    
    # Create horizontal gradient from red to blue
    for x in range(width):
        for y in range(height):
            red = int(255 * (1 - x / width))
            blue = int(255 * (x / width))
            image.putpixel((x, y), (red, 0, blue))
    
    image_path = temp_dir / "gradient_image.png"
    image.save(image_path)
    return image_path


@pytest.fixture
def monochrome_image(temp_dir):
    """Create a monochrome test image."""
    width, height = 100, 100
    image = Image.new('RGB', (width, height), color=(128, 128, 128))
    
    image_path = temp_dir / "monochrome_image.png"
    image.save(image_path)
    return image_path


@pytest.fixture
def complex_image(temp_dir):
    """Create a complex test image with many colors."""
    width, height = 300, 300
    image = Image.new('RGB', (width, height))
    pixels = []
    
    # Generate random colors
    np.random.seed(42)  # For reproducible tests
    for y in range(height):
        row = []
        for x in range(width):
            # Create some pattern to ensure we get varied colors
            r = int(255 * abs(np.sin(x * 0.01)) * abs(np.sin(y * 0.01)))
            g = int(255 * abs(np.cos(x * 0.01)) * abs(np.cos(y * 0.01)))
            b = int(255 * abs(np.sin(x * 0.01)) * abs(np.cos(y * 0.01)))
            row.append((r, g, b))
        pixels.append(row)
    
    # Set pixels
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), pixels[y][x])
    
    image_path = temp_dir / "complex_image.png"
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_colors():
    """Create sample ColorInfo objects for testing."""
    return [
        ColorInfo(
            rgb=(255, 0, 0),
            hex="#ff0000",
            name="red",
            frequency=1000,
            percentage=25.0,
            hsv=(0.0, 1.0, 1.0)
        ),
        ColorInfo(
            rgb=(0, 255, 0),
            hex="#00ff00", 
            name="lime",
            frequency=1000,
            percentage=25.0,
            hsv=(0.33, 1.0, 1.0)
        ),
        ColorInfo(
            rgb=(0, 0, 255),
            hex="#0000ff",
            name="blue",
            frequency=1000,
            percentage=25.0,
            hsv=(0.67, 1.0, 1.0)
        ),
        ColorInfo(
            rgb=(255, 255, 0),
            hex="#ffff00",
            name="yellow",
            frequency=1000,
            percentage=25.0,
            hsv=(0.17, 1.0, 1.0)
        )
    ]


@pytest.fixture
def sample_color_data(sample_colors):
    """Create sample ColorData for testing."""
    metadata = ImageMetadata(
        path="test_image.png",
        filename="test_image.png",
        size=(200, 200),
        format="PNG",
        mode="RGB",
        total_pixels=40000,
        file_size=1024
    )
    
    return ColorData(
        image_metadata=metadata,
        colors=sample_colors,
        unique_color_count=4,
        dominant_color=sample_colors[0],
        average_color=sample_colors[0]
    )


@pytest.fixture
def multiple_images(temp_dir):
    """Create multiple test images for batch testing."""
    images = []
    
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        image = Image.new('RGB', (100, 100), color=color)
        image_path = temp_dir / f"image_{i}.png"
        image.save(image_path)
        images.append(image_path)
    
    return images


@pytest.fixture(scope="session")
def algorithms():
    """List of all supported algorithms."""
    return ["kmeans", "dbscan", "median_cut", "octree", "histogram"]


@pytest.fixture(scope="session") 
def export_formats():
    """List of export formats to test."""
    return ["json", "csv", "css", "scss", "html", "svg"]
