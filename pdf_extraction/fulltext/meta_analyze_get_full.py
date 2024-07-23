#==============================================================================
# For the meta analysis and database: 
# This is STEP 3 in the pipeline: 
# Using the retrieved citations based on keyword search via PubMed, try to pull
# full texts from open access sources. 
# 
# Automate creation (if needed) and uploading of Methods and Results sections 
# from the full text to the labeling project in Label Studio (currently mbb_abstracts). 
# Although I am not currently pursuing this, this could be used to train a  
# second custom NER for the full text.
# 
# Use the current version of the NER to streamline labeling by generating 
# predictions. The labeling process is iterative! Label, predict, correct, 
# generate a new version of the NER (via meta_analyze_model_update.py), use it 
# to label, predict...
#
# Predicting labels needs model_abstract_app.py to be running on a terminal! 
#
# export NCBI_API_KEY="f2857b2abca3fe365c756aeb647e06417b08"
#==============================================================================
#Libraries
NCBI_API_KEY = "f2857b2abca3fe365c756aeb647e06417b08"	

#libraries
import pandas as pd

#For references 
import os
os.environ['NCBI_API_KEY'] = NCBI_API_KEY

#For label studio
from label_studio_sdk import Client

#NLP
import spacy

#The shared custom definitions
#NOTE: This line might have to be modified as structure changes and 
#we move towards deployment
## Add the project root directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath('./../'))  
import common.utilities
#==============================================================================
#Fetch records from PubMed
#==============================================================================
fetcher = PubMedFetcher(cachedir='./papers/')
# Construct search query: based on Averil et al 2022
#query = '(mycorrhiz*) AND ((soil inocul*) OR (whole soil inocul*) OR (soil transplant*) OR (whole community transplant*))'

#Modified to be more specific: 
query = '(mycorrhiz*) AND ((soil inocul*) OR (whole soil inocul*) OR (soil transplant*) OR (whole community transplant*)) AND biomass NOT review'
#(mycorrhiz*) AND ((soil inocul*) OR (whole soil inocul*) OR (soil transplant*) OR (whole community transplant*)) AND biomass AND (control OR non-inoculate* OR non inoculate* OR uninoculate* OR steril* OR noncondition* OR uncondition* OR non condition*) NOT review
# Use the fetcher to get PMIDs for the query
pmids = fetcher.pmids_for_query(query,retmax = 10000)

# Create an empty list to store articles
articles = []

# Get the information for each article: 
#C:\Users\jusin\.cache\findit.db
for pmid in pmids:
    article = fetcher.article_by_pmid(pmid)
    articles.append(article)

#==============================================================================
#Try to get full text PDFs from NCBI
#==============================================================================
# Directory to save the full text articles
save_directory = './../papers/'

# Fetch and save full text articles
common.utilities.get_full_text(articles, save_directory)

#==============================================================================
# Create project to fine-tune an NER to pull useful info from abstracts
# 1. First, link to Label Studio to label text
#==============================================================================
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '45d69e3e9c859f4583dd42e5246f346e509a0a8e'
PROJECT_ID = '1' #This links to the full text trainer

#==============================================================================
# If needed: Define the labeling configuration XML and update
#==============================================================================

label_config_xml = """
<View>
  <Labels name="label" toName="text">
    <Label value="TREATMENT" background="#ff0000"/>
    <Label value="INOCTYPE" background="#00ffff"/>
    <Label value="SOILTYPE" background="#ff9900"/>
    <Label value="FIELDGREENHOUSE" background="#FFA500"/>
    <Label value="LANDUSE" background="#800080"/>
    <Label value="RESPONSE" background="#00ff00"/>
    <Label value="ECOTYPE" background="#0000ff"/>
    <Label value="ECOREGION" background="#ffff00"/>
    <Label value="LOCATION" background="#A133FF"/>
    <Label value="CARDINAL" background="#FF5733"/>
    <Label value="PERCENTAGE" background="#D4380D"/>
    <Label value="UNITS" background="#33FFA1"/></Labels>
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
# This section only needs to be run the first time, or 
# if new papers are added:
# 1. Import the PDFs
# 2. Preprocess text with NLTK
# 3. Extract Methods and Results sections 
# 4. Upload these to label studio
#==============================================================================
#Step 1: Extract Text from PDF
#Step 2: Preprocess Text
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
    'conclusion': 'discussion',
    'summary': 'discussion'
}
#Step 4: Upload text data to Label Studio

# Now put all of this together and do it! 
# Keep track of PDFs that have already been processed
record_file = 'processed_pdfs.txt'

# Ensure the record file exists - if this is the first run, e.g.
if not os.path.exists(record_file):
    with open(record_file, 'w') as f:
        pass

# Load processed PDFs from the record file
with open(record_file, 'r') as f:
    processed_pdfs = set(f.read().splitlines())

# Get the list of current PDFs in the directory
current_pdfs = {f for f in os.listdir(save_directory) if f.endswith('.pdf')}

# Identify new PDFs that have not been processed
new_pdfs = current_pdfs - processed_pdfs

# Process the new PDFs
for pdf in new_pdfs:
    #1.
    pdf_path = "./../papers/" + pdf
    pdf_text = common.utilities.extract_text_from_pdf(pdf_path)
    #2. 
    sentences = common.utilities.preprocess_text(pdf_text)
    #3.
    sections = common.utilities.identify_sections(sentences,section_mapping)
    #4.
    #Get the methods and upload
    methods_text = " ".join(sections.get('methods', [])) 
    if methods_text: # Check it exists
        common.utilities.upload_task(methods_text, PROJECT_ID)
    else:
        print(f"No Methods found for article with PMID: {pdf}")
    #Get the reults and upload
    results_text = " ".join(sections.get('results', []))
    if results_text: # Check it exists
        common.utilities.upload_task(results_text, PROJECT_ID)
    else:
        print(f"No Results found for article with PMID: {pdf}")
    #Keep track of processed PDFs
    processed_pdfs.add(pdf)

# Update the record file with the new processed PDFs
with open(record_file, 'w') as f:
    for pdf in processed_pdfs:
        f.write(f"{pdf}\n")

#==============================================================================
#Link a custom NER model to Label Studio and generate suggested labels. 
#Turn this into a loop to get better predictions as more annotations are added. 
#==============================================================================
#For label studio
import requests
from label_studio_sdk import Client

LABEL_STUDIO_URL = 'http://localhost:8080' #Run with model_abstract_app.py
API_KEY = '45d69e3e9c859f4583dd42e5246f346e509a0a8e'
PROJECT_ID = '1' #This links to the abstract-specific trainer

# Initialize the Label Studio client
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

# Get the project
project = ls.get_project(PROJECT_ID)

# Fetch tasks from the project
tasks = project.get_tasks()

# Filter out completed tasks
def is_task_completed(task):
    return len(task['annotations']) > 0  # Adjust this condition based on your project's definition of "completed"

incomplete_tasks = [task for task in tasks if not is_task_completed(task)]

# Prepare a list to hold the predictions
predictions = []

# Process the first 20 incomplete tasks
# Make sure the model is being hosted! 
# py -3.10 model_abstract_app.py
for task in incomplete_tasks[:20]:
    text = task['data']['text']  # Adjust this key based on your data format
    response = requests.post('http://localhost:5000/predict', json={'text': text})
    predictions_response = response.json()
    # Prepare predictions in Label Studio format
    annotations = [{
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": pred['start'],
            "end": pred['end'],
            "labels": [pred['label']]
        }

    } for pred in predictions_response]
    # Append the prediction to the list
    predictions.append({
        'task': task['id'],
        'result': annotations, 
        'model_version': 'custom_web_ner_abs_v381'  # You can set this to track the version of your model
    })

# Create predictions in bulk
project.create_predictions(predictions)

# #saving
# with open('predictions_v381','wb') as f:
#     pickle.dump(predictions,f)

# # Loading a variable
# with open('predictions_v2', 'rb') as f:
#     loaded_variable = pickle.load(f)

# print(loaded_variable)


