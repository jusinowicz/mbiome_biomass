#==============================================================================
# We need to train NER for the microbiome-biomass database expansion. 
# Implemente a labeling and training workflow using Label Studio to starty 
# labeling relevant text from the Methods and Results sections of PDFs. 
#
#
# py -3.10 -m pip install label-studio
#==============================================================================
py -3.10 

import requests
import os
#PDF extraction
import fitz  # PyMuPDF
#Text preprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy
# Load pre-trained SciBERT model from spaCy
nlp = spacy.load("en_core_sci_md")

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '45d69e3e9c859f4583dd42e5246f346e509a0a8e' # Obtain this from the Label Studio UI
PROJECT_ID = '1'

#==============================================================================
#Print Available Projects: List all projects to verify the project ID.
#==============================================================================

response = requests.get(
	f'{LABEL_STUDIO_URL}/api/projects',
	headers={'Authorization': f'Token {API_KEY}'}
	)
print("Available projects:", response.json())


#==============================================================================
# If needed: Define the labeling configuration XML and update
#==============================================================================

label_config_xml = """
<View>
  <Labels name="label" toName="text">
    <Label value="TREATMENT" background="#ff0000"/>
    <Label value="RESPONSE" background="#00ff00"/>
    <Label value="ECOTYPE" background="#0000ff"/>
    <Label value="ECOREGION" background="#ffff00"/>
    <Label value="LOCATION" background="#ff00ff"/>
    <Label value="LAT" background="#00ffff"/>
    <Label value="LON" background="#ff9900"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
"""

# Update the project with the new labeling configuration
response = requests.patch(
	f'{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}',
	headers={'Authorization': f'Token {API_KEY}', 'Content-Type': 'application/json'},
	json={'label_config': label_config_xml}
)

print("Status Code:", response.status_code)
print("Response Text:", response.text)

#==============================================================================
#Create a list of dictionaries with entries that represent Methods and Results 
#==============================================================================
#Step 1: Extract Text from PDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

pdf_path = './papers/Semchenko2019.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

#Step 2: Preprocess Text
# Download NLTK data files
nltk.download('punkt')

def preprocess_text(text):
    # Remove References section
    text = re.sub(r'\bREFERENCES\b.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    return sentences

sentences = preprocess_text(pdf_text)

#Step 3: Identify Sections

#Define a mapping for section variations
section_mapping = {
    'abstract': 'abstract',
    'introduction': 'introduction',
    'background': 'introduction',
    'methods': 'methods',
    'materials and methods': 'methods',
    'methodology': 'methods',
    'experimental methods': 'methods',
    'results': 'results',
    'findings': 'results',
    'discussion': 'discussion',
    'conclusion': 'conclusion'
}

def identify_sections(sentences):
	sections = {}
	current_section = None
    
	# Enhanced regex to match section headers
	section_header_pattern = re.compile(r'\b(Abstract|Introduction|Methods|Materials and Methods|Results|Discussion|Conclusion|Background|Summary|Acknowledgments|References|Bibliography)\b', re.IGNORECASE)
	for sentence in sentences:
		# Check if the sentence is a section header
		header_match = section_header_pattern.search(sentence)
		if header_match:
			section_name = header_match.group(1).lower()
			normalized_section = section_mapping.get(section_name, section_name)
			current_section = normalized_section
			sections[current_section] = []
			print(f"Matched Section Header: {header_match}")  # Debugging line
		elif current_section:
			sections[current_section].append(sentence)
	return sections

sections = identify_sections(sentences)

#==============================================================================
# Upload text data to Label Studio
#==============================================================================
# Function to upload text data to Label Studio

def upload_task(text, project_id):
	import_url = f'{LABEL_STUDIO_URL}/api/projects/{project_id}/import'
	print("Import URL:", import_url)
	response = requests.post(
		import_url,
		headers={'Authorization': f'Token {API_KEY}'},
		json=[{
			'data': {
				'text': text
			}
		}]
	)
	print("Status Code:", response.status_code)
	print("Response Text:", response.text)
	try:
		response_json = response.json()
		print(response_json)
		return response_json
	except requests.exceptions.JSONDecodeError as e:
		print("Failed to decode JSON:", e)
		return None

#==============================================================================
#Run code on directory 
#==============================================================================

# Directory containing the PDFs
pdf_dir = './papers/'

# Loop through each PDF in the directory
for pdf_filename in os.listdir(pdf_dir):
	if pdf_filename.endswith('.pdf'):
		pdf_path = os.path.join(pdf_dir, pdf_filename)
		pdf_text = extract_text_from_pdf(pdf_path)
		sentences = preprocess_text(pdf_text)
		sections = identify_sections(sentences)
		# Upload Methods and Results sections
		if 'methods' in sections:
			methods_text = ' '.join(sections['methods'])
			upload_task(methods_text, PROJECT_ID)
		if 'results' in sections:
			results_text = ' '.join(sections['results'])
			upload_task(results_text, PROJECT_ID)

print("All tasks uploaded successfully!")
