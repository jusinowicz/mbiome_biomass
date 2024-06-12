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
import fitz  # PyMuPDF
import os
import re
import nltk
import requests
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

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
# If needed: # Create a new project (or start locally with label-studio start 
# and use GUI)
#==============================================================================
 
#response = requests.post(
#     f'{LABEL_STUDIO_URL}/api/projects',
#     headers={'Authorization': f'Token {API_KEY}'},
#     json={
#         'title': 'NER Annotation Project',
#         'label_config': '''
#         <View>
#           <Labels name="label" toName="text">
#             <Label value="TREATMENT" background="#ffcc00"/>
#             <Label value="RESPONSE" background="#00ffcc"/>
#           </Labels>
#           <Text name="text" value="$text"/>
#         </View>
#         '''
#     }
# )
# project = response.json()
# project_id = project['id']
# print(f"Created project with ID: {project_id}")

#==============================================================================
# If needed: Define the labeling configuration XML and update
#==============================================================================

label_config_xml = """
<View>
	<Labels name="label" toName="text">
		<Label value="TREATMENT" background="#ff0000"/>
		<Label value="RESPONSE" background="#00ff00"/>
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

# pdf_data = [
# 	{
#         "methods": "The experiment involved growing plants in inoculated and uninoculated soils.",
#         "results": "The response was measured in terms of plant biomass."
#     },
#     {
#         "methods": "Soil samples were taken from natural environments and used to inoculate sterile soil.",
#         "results": "Plant growth in inoculated soil was significantly higher compared to non-inoculated soil."
#     }
#     # Add more PDF data as needed
# ]

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


# Function to upload text data to Label Studio: THE DEBUG VERSION
# def upload_task(text, project_id):
# 	response = requests.post(
# 		f'{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import',
# 		headers={'Authorization': f'Token {API_KEY}'},
# 		json=[{
# 			'data': {
# 				'text': text
# 			}
# 		}]
# 	)
# 	# Print the response for debugging
# 	print("Status Code:", response.status_code)
# 	print("Response Text:", response.text)
# 	# Check if the response is valid JSON
# 	try:
# 		response_json = response.json()
# 		return response_json
# 	except requests.exceptions.JSONDecodeError as e:
# 		print("Failed to decode JSON:", e)
# 		return None


# Loop through each PDF and upload Methods and Results sections
# for pdf in pdf_data:
# 	methods_section = pdf["methods"]
# 	results_section = pdf["results"]
# 	upload_task(methods_section, PROJECT_ID)
# 	upload_task(results_section, PROJECT_ID)

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
