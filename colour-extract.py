import sys
import colorsys
from PIL import Image, ImageDraw
import webcolors
from colormath.color_objects import sRGBColor
from datetime import datetime
from tkinter import Tk, filedialog
from tqdm import tqdm
 
 
def calculate_color_difference(color1, color2):
    """
    Calculates the perceptual difference between two color objects
    using the Euclidean distance between their RGB value tuples.
    
    This works by:
    1. Extracting the RGB value tuples for each color object
    2. Calculating the difference between the R, G, and B values
    3. Squaring the differences
    4. Summing the squared differences
    5. Taking the square root of the sum
    
    This gives a measure of the total geometric distance between the 
    two colors in RGB space, which corresponds to their visual difference.
    
    Args:
      color1: First color object
      color2: Second color object
      
    Returns:
      Float value representing the perceptual distance between the colors
    """
    
    r1, g1, b1 = color1.get_value_tuple()
    r2, g2, b2 = color2.get_value_tuple()
 
    r_diff = r1 - r2
    g_diff = g1 - g2
    b_diff = b1 - b2
 
    return ((r_diff ** 2) + (g_diff ** 2) + (b_diff ** 2)) ** 0.5
 
 
 
def generate_color_palette(colors, palette_size):
    # Generates a color palette image from a list of RGB color tuples.
    #
    # Args:
    #   colors: List of RGB color tuples.
    #   palette_size: Number of colors to include in the palette.
    # 
    # Returns:
    #   Image object containing the generated color palette.
    swatch_size = 100
    palette_width = palette_size * swatch_size
    palette_height = swatch_size
    palette_image = Image.new("RGB", (palette_width, palette_height))
    draw = ImageDraw.Draw(palette_image)
 
    x = 0
    for color in colors:
        draw.rectangle([(x, 0), (x + swatch_size, palette_height)], fill=color, outline="black")
        x += swatch_size
 
    return palette_image
 
 
# Extracts detailed color information from the given image file 
# and saves it to a text file and color palette image.
#
# Args:
#   image_path: Path to image file to extract colors from.
#   
# The function opens the image, loops through pixels to collect 
# color data, calculates averages and distributions, finds closest  
# CSS color names, generates a color palette image, and saves the
# output to a text file and image file.
def extract_color_information(image_path):
    """Extracts detailed color information from the given image file and saves it to a text file and color palette image.
    
    Args:
      image_path: Path to image file to extract colors from.
      
    The function opens the image, loops through pixels to collect color data, calculates averages and distributions, finds closest CSS color names, generates a color palette image, and saves the output to a text file and image file.
    """
    try:
        # Open and convert the image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
 
        # Extract color information
        colors, color_counts = set(), {}
        for x in tqdm(range(width), desc="Extracting colors", unit="column"):
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                color = (r, g, b)
                colors.add(color)
                color_counts[color] = color_counts.get(color, 0) + 1
 
        # Calculate average color and convert to HSV
        average_color = image.resize((1, 1)).getpixel((0, 0))
        average_color_hsv = colorsys.rgb_to_hsv(*average_color)
 
        # Extract dominant color
        dominant_color = max(color_counts, key=color_counts.get)
 
        # Calculate color distribution
        total_pixels = width * height
        color_distribution = {color: count / total_pixels * 100 for color, count in color_counts.items()}
 
        # Find color names and HEX codes
        # Loops through each extracted color from the image 
        # Uses webcolors module to find the closest CSS color name and HEX code
        # Appends the name and code to separate lists
        # Catches any errors and skips invalid colors
        color_names, color_hex_codes = [], []
        for color in tqdm(colors, desc="Finding color names", unit="color"):
            try:
                name = webcolors.rgb_to_name(color)
                hex_code = webcolors.rgb_to_hex(color)
                color_names.append(name)
                color_hex_codes.append(hex_code)
            except ValueError:
                pass
 
        # Find similar colors
        # Looks up the closest predefined CSS color for each extracted color 
        # and stores the result in a dictionary mapping colors to their 
        # closest predefined CSS color name
        predefined_colors = webcolors.CSS3_NAMES_TO_HEX.values()
        similar_colors = {min(predefined_colors, key=lambda x: calculate_color_difference(
            sRGBColor(color[0] / 255, color[1] / 255, color[2] / 255),
            sRGBColor.new_from_rgb_hex(x)
        )) for color in tqdm(colors, desc="Finding similar colors", unit="color")}
 
        # Save color information to a text file
        output_file = f"{image_path.rsplit(&#39;.&#39;, 1)[0]}.txt"
        """Writes extracted color information to output text file.
        
        The output file path is generated from the input image path. 
        The file contains the following information:
        
        - Image path
        - Image dimensions 
        - Number of unique colors
        - Average color in RGB and HSV
        - Dominant color
        - Color distribution percentages 
        - Color names and hex codes
        - Similar colors mapped to closest CSS color names
        """
        with open(output_file, "w") as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Size: {width}x{height}\n")
            f.write(f"Number of unique colors: {len(colors)}\n")
            f.write(f"Average color (RGB): {average_color}\n")
            f.write(f"Average color (HSV): {average_color_hsv}\n")
            f.write(f"Dominant color: {dominant_color}\n")
            f.write("Color Distribution:\n")
            for color, percentage in color_distribution.items():
                f.write(f"{color}: {percentage:.2f}%\n")
            f.write("Color Names and HEX Codes:\n")
            for name, hex_code in zip(color_names, color_hex_codes):
                f.write(f"{name} (HEX: {hex_code})\n")
            f.write("Similar Colors:\n")
            for color in similar_colors:
                f.write(color + "\n")
 
        # Generate color palette image
        # Saves the generated color palette image to disk and prints completion messages.
        # The color palette image path is generated from the input image path with a _palette suffix.  
        # Completion messages indicate where the color information text file and palette image were saved.
        palette_image = generate_color_palette(list(colors), palette_size=10)
        palette_image_path = f"{image_path.rsplit(&#39;.&#39;, 1)[0]}_palette.png"
        palette_image.save(palette_image_path)
 
        # Print completion message
        print(f"Color information saved to {output_file}")
        print(f"Color palette image saved to {palette_image_path}")
 
    except Exception as e:
        # Logs error details to a log file when an exception occurs.
        #
        # The current timestamp, error message, and stack trace are written 
        # to the error.log file. This allows errors to be tracked over time
        # for debugging.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = str(e)
        with open("error.log", "a") as error_file:
            error_file.write(f"[{timestamp}] An error occurred: {error_message}\n")
 
 
def prompt_for_image():
    # Prompts the user to select an image file using a file dialog, 
    # then extracts color information from the selected image.
    #
    # Opens a file dialog for the user to select an image file. The 
    # allowed file types are PNG, JPG, and JPEG images. 
    #
    # If an image is selected, calls extract_color_information() on 
    # the path to extract colors.
    #
    # If no image is selected, prints a message indicating no image 
    # was chosen.
    #
    # No return value. Prompts the user and processes the selected 
    # image file if one is chosen.
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an image file",
                                            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
    if image_path:
        extract_color_information(image_path)
    else:
        print("No image file selected.")
 
 
if __name__ == "__main__":
    # If run as a script, check command line arguments for an image path.  
    # If provided, extract colors from that image.
    # Otherwise, prompt the user to select an image file with a dialog.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        extract_color_information(image_path)
    else:
        prompt_for_image()
