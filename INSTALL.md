# 📦 PyColour Extract 2.0 - Installation Guide

This guide will help you install and set up PyColour Extract 2.0 on your system.

## 📋 Prerequisites

Before installing, make sure you have:

- **Python 3.8 or higher** ([Download Python](https://python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **pip** (usually comes with Python)

### Verify Prerequisites

```bash
# Check Python version (should be 3.8+)
python --version

# Check Git installation
git --version

# Check pip
pip --version
```

## 🚀 Quick Installation

### Method 1: Direct GitHub Installation (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git

# Test the installation
pycolour-extract --version
```

### Method 2: Clone and Install (For Development)

```bash
# Clone the repository
git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git

# Navigate to the project directory
cd Python-Colour-Info-Extractor

# Install in development mode
pip install -e .

# Test the installation
pycolour-extract --version
```

## 🔧 Installation Options

### Basic Installation

```bash
pip install git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git
```

This installs the core functionality with all required dependencies.

### Full Installation with Optional Features

```bash
# Clone first
git clone https://github.com/Kynlos/Python-Colour-Info-Extractor.git
cd Python-Colour-Info-Extractor

# Install with all optional dependencies
pip install -e ".[gui,export,dev]"
```

**Optional dependency groups:**
- `gui`: GUI components (future)
- `export`: Advanced export formats (Excel, additional formats)
- `dev`: Development tools (testing, linting, pre-commit hooks)

## 🐍 Virtual Environment (Recommended)

Using a virtual environment is recommended to avoid conflicts:

### Windows

```cmd
# Create virtual environment
python -m venv pycolour-env

# Activate virtual environment
pycolour-env\Scripts\activate

# Install PyColour Extract
pip install git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git

# Test installation
pycolour-extract --version
```

### macOS/Linux

```bash
# Create virtual environment
python -m venv pycolour-env

# Activate virtual environment
source pycolour-env/bin/activate

# Install PyColour Extract
pip install git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git

# Test installation
pycolour-extract --version
```

## ✅ Verify Installation

After installation, test that everything works:

```bash
# Check version
pycolour-extract --version

# Get help
pycolour-extract --help

# Test with a sample image (if you have one)
pycolour-extract path/to/your/image.jpg
```

## 🔄 Updating

To update to the latest version:

### If installed via pip:
```bash
pip install --upgrade git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git
```

### If installed via clone:
```bash
cd Python-Colour-Info-Extractor
git pull origin main
pip install -e .
```

## 🗑️ Uninstallation

To completely remove PyColour Extract:

```bash
# Uninstall the package
pip uninstall pycolour-extract

# If you cloned the repository, you can also delete the folder
rm -rf Python-Colour-Info-Extractor  # Linux/macOS
# or
rmdir /s Python-Colour-Info-Extractor  # Windows
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Command not found: pycolour-extract"

**Solution:**
- Make sure pip install completed successfully
- Try `python -m pycolour_extract.cli` instead
- Check if the Scripts directory is in your PATH

#### 2. "No module named 'pycolour_extract'"

**Solution:**
- Reinstall the package: `pip install --force-reinstall git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git`
- Make sure you're using the correct Python environment

#### 3. Permission denied errors (Linux/macOS)

**Solution:**
```bash
# Install for current user only
pip install --user git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git
```

#### 4. SSL/Git errors

**Solution:**
```bash
# Download as ZIP and install locally
# 1. Download ZIP from GitHub
# 2. Extract the archive
# 3. Navigate to the extracted folder
# 4. Run: pip install .
```

#### 5. Dependency conflicts

**Solution:**
- Use a virtual environment (recommended)
- Or install with: `pip install --no-deps git+https://github.com/Kynlos/Python-Colour-Info-Extractor.git`
- Then manually install dependencies

### Getting Help

If you encounter issues:

1. **Check the [GitHub Issues](https://github.com/Kynlos/Python-Colour-Info-Extractor/issues)**
2. **Create a new issue** with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Installation method used

## 🎯 Quick Start After Installation

Once installed, try these commands:

```bash
# Get help and see all options
pycolour-extract --help

# Basic extraction
pycolour-extract your-image.jpg

# Advanced analysis
pycolour-extract image.jpg --algorithm kmeans --export-formats json css html --analyze-emotion --analyze-brand

# Batch processing
pycolour-extract batch ./image-folder --recursive

# Compare images
pycolour-extract compare image1.jpg image2.jpg

# Generate color suggestions
pycolour-extract suggest "#FF0000" "#00FF00" --suggestion-type harmony
```

Enjoy using PyColour Extract 2.0! 🎨
