#==============================================================================
# For the meta analysis and database: 
# This is STEP 1 and STEP1C in the pipeline: 
# STEP1: Automate identification and retrieval of citations, abstracts for potentially 
# relevant papers based on keyword search via PubMed. 
# 
# Automate creation (if needed) and uploading of abstracts to the labeling 
# project in Label Studio (currently mbb_abstracts). This is done to help train
# the custom NER.
# 
# STEP1C: Use the current version of the NER to streamline labeling by generating 
# predictions. The labeling process is iterative! Label, predict, correct, 
# generate a new version of the NER (via meta_analyze_model_update.py), use it 
# to label, predict...
#
# Predicting labels needs model_abstrac_app.py to be running on a terminal! 
#
# export NCBI_API_KEY="f2857b2abca3fe365c756aeb647e06417b08"
#==============================================================================
#Libraries
NCBI_API_KEY = "f2857b2abca3fe365c756aeb647e06417b08"	

#libraries
import pandas as pd
import dill
import lxml.etree as ET
import csv


#To get references from NCBI:
from metapub import PubMedFetcher 
import os
os.environ['NCBI_API_KEY'] = NCBI_API_KEY

#For label studio
from label_studio_sdk import Client

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
for pmid in pmids:
    article = fetcher.article_by_pmid(pmid)
    articles.append(article)


#==============================================================================
# If you need to export DOIs, run the following: 
#==============================================================================
# Create an empty list to store DOIs and PMIDs
doi_pmid_pairs = []

# Loop through the articles and extract DOIs and PMIDs
for article in articles:
    doi = article.doi
    pmid = article.pmid
    if doi and pmid:
        doi_pmid_pairs.append((doi, pmid))

# # Save the DOIs and PMIDs to a CSV file
with open("./../fulltext/all_DOIs.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['DOI', 'PMID'])  # Write header
    for doi, pmid in doi_pmid_pairs:
        csvwriter.writerow([doi, pmid])


#==============================================================================
# Create project to fine-tune an NER to pull useful info from abstracts
# 1. First, link to Label Studio to label text
#==============================================================================
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '45d69e3e9c859f4583dd42e5246f346e509a0a8e'
PROJECT_ID = '2' #This links to the abstract-specific trainer


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
#Run code on directory 
#==============================================================================
# Loop through each abstract
for a in articles:
	# Upload abstracts
	abstract = a.abstract
	if abstract: # Check if the article has an abstract
		common.utilities.upload_task(abstract, PROJECT_ID)
	else:
		print(f"No abstract found for article with PMID: {article.pmid}")

#==============================================================================
#Link a custom NER model to Label Studio and generate suggested labels. 
#Turn this into a loop to get better predictions as more annotations are added. 
#==============================================================================

LABEL_STUDIO_URL = 'http://localhost:8080' #Run with model_abstract_app.py
API_KEY = '45d69e3e9c859f4583dd42e5246f346e509a0a8e'
PROJECT_ID = '2' #This links to the abstract-specific trainer

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

# Process the e.g. first 20 [:20] incomplete tasks
# Make sure the model is being hosted! 
# py -3.10 model_abstract_app.py
for task in incomplete_tasks[:50]:
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