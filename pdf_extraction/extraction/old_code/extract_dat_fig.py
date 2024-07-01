#==============================================================================
# This was meant to be a porgram to reliably extract data from figures in 
# scientific papers. It has mostly been a failure. The main problem is that 
# extraction only really seems possible when figures are rasters. But the 
# majority of scientific publications use vector formats. While it seems 
# plausible to make a piece of the pipeline conversion from vector to raster,
# it actually seems really challenging to extract a vector image from a PDF
# reliably. 
#
# Anyway, tabling this for now and shifting to focus on text and table extraction.
# As other programs and websites like WebPlotDigitizer seem more reliable. 
#==============================================================================

#Load libraries 
import pdfplumber
import re
import spacy
import os
import tabula
import pandas as pd
import fitz  # PyMuPDF
import cv2
import pytesseract
import io
from PIL import Image
from bs4 import BeautifulSoup
from pdf2image import convert_from_path


# Output directory for the extracted images
output_dir = "./images/"
# Desired output image format
output_format = "png"
# Minimum width and height for extracted images
min_width = 100
min_height = 100
# Create the output directory if it does not exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# file path you want to extract images from
file = "./papers/skarssonHeyser2015.pdf"
# open the file
pdf_file = fitz.open(file)

page_number = 8
pages = convert_from_path(file)
image_path = 'page_image.png'
pages[page_number].save(image_path, 'PNG')
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ocr_result = pytesseract.image_to_string(gray)

# Iterate over PDF pages
for page_index in range(len(pdf_file)):
	# Get the page itself
	page = pdf_file[page_index]
	# Get image list
	image_list = page.get_images(full=True)
	# Print the number of images found on this page
	if image_list:
		print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
	else:
		print(f"[!] No images found on page {page_index}")
	# Iterate over the images on the page
	for image_index, img in enumerate(image_list, start=1):
		# Get the XREF of the image
		xref = img[0]
        # Extract the image bytes
		base_image = pdf_file.extract_image(xref)
		image_bytes = base_image["image"]
		# Get the image extension
		image_ext = base_image["ext"]
		# Load it to PIL
		image = Image.open(io.BytesIO(image_bytes))
		# Check if the image meets the minimum dimensions and save it
		if image.width >= min_width and image.height >= min_height:
			image.save(
				open(os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}"), "wb"),
				format=output_format.upper())
		else:
			print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")
	# Extract vector graphics (drawings) as a single SVG file
	svg_data = page.get_svg_image()
	svg_file_path = os.path.join(output_dir, f"page{page_index + 1}.svg")
	with open(svg_file_path, "w") as svg_file:
		svg_file.write(svg_data)
	print(f"[+] Saved SVG for page {page_index + 1}.")


# Define the directory containing SVG files
svg_dir = output_dir
svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]

# Function to extract figures from an SVG file
def extract_figures_from_svg(svg_file):
	with open(svg_file, 'r', encoding='utf-8') as file:
		svg = file.read()
	soup = BeautifulSoup(svg, 'xml')
    
	# Find elements that correspond to figures (e.g., <g> tags)
	figures = soup.find_all('g')
    
	# Extract each figure as a separate SVG file
	for i, figure in enumerate(figures):
		figure_svg = str(figure)
		output_file = f"{os.path.splitext(svg_file)[0]}_figure_{i+1}.svg"
		with open(output_file, 'w',encoding='utf-8') as output:
			output.write(figure_svg)

# Apply the extraction function to all SVG files
for svg_file in svg_files:
	extract_figures_from_svg(os.path.join(svg_dir, svg_file))

# List the extracted figure files
extracted_files = [f for f in os.listdir(svg_dir) if 'figure_' in f and f.endswith('.svg')]
print(extracted_files)


	# # Extract vector graphics (drawings)
	# drawing_list = page.get_drawings()
	# if drawing_list:
	# 	print(f"[+] Found a total of {len(drawing_list)} vector graphics in page {page_index}")
	# else:
	# 	print(f"[!] No vector graphics found on page {page_index}")
	# for drawing_index, drawing in enumerate(drawing_list, start=1):
	# 	# Extract the drawing (vector graphic) and save as SVG
	# 	svg_data = page.get_svg_image(drawing)
	# 	svg_file_path = os.path.join(output_dir, f"vector{page_index + 1}_{drawing_index}.svg")
	# 	with open(svg_file_path, "w") as svg_file:
	# 		svg_file.write(svg_data)