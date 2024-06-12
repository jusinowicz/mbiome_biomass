#==============================================================================
# For the meta analysis and database: Try to automate retrieval of 
# identification of relevant papers and retrieval of paper infos and text as 
# much as possible. 
# export NCBI_API_KEY="f2857b2abca3fe365c756aeb647e06417b08"
#==============================================================================
#Libraries
NCBI_API_KEY = "f2857b2abca3fe365c756aeb647e06417b08"	

#libraries
import pandas as pd
from metapub import PubMedFetcher 
import os
os.environ['NCBI_API_KEY'] = NCBI_API_KEY
#==============================================================================
#Fetch records from PubMed
#==============================================================================
fetcher = PubMedFetcher()

# Construct search query: based on Averil et al 2022 
query = '(mycorrhiz*) AND ((soil inocul*) OR (whole soil inocul*) OR (soil transplant*) OR (whole community transplant*)) AND biomass NOT review'
#(mycorrhiz*) AND ((soil inocul*) OR (whole soil inocul*) OR (soil transplant*) OR (whole community transplant*)) AND biomass AND (control OR non-inoculate* OR non inoculate* OR uninoculate* OR steril* OR noncondition* OR uncondition* OR non condition*) NOT review
# Use the fetcher to get PMIDs for the query
pmids = fetcher.pmids_for_query(query)

# Create an empty list to store articles
articles = []

# Get the information for each article: 
for pmid in pmids:
    article = fetcher.article_by_pmid(pmid)
    articles.append(article)

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
    <Label value="LOCATION" background="#ff00ff"/>
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
# Loop through each abstract
for a in articles:
	# Upload abstracts
	abstract = a.abstract
	if abstract: # Check if the article has an abstract
		upload_task(abstract, PROJECT_ID)
	else:
		print(f"No abstract found for article with PMID: {article.pmid}")

#==============================================================================
#Link a custom NER model to Label Studio and generate suggested labels. 
#Turn this into a loop to get better predictions as more annotations are added. 
#==============================================================================

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
        'model_version': 'custom_sci_ner_abs'  # You can set this to track the version of your model
    })

# Create predictions in bulk
project.create_predictions(predictions)