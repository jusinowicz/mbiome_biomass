#==============================================================================
# For the meta analysis and database: Try to automate retrieval of 
# identification of relevant papers and retrieval of paper infos and text as 
# much as possible. 
# export NCBI_API_KEY="f2857b2abca3fe365c756aeb647e06417b08"
#==============================================================================
NCBI_API_KEY = "f2857b2abca3fe365c756aeb647e06417b08"	

#libraries
import pandas as pd
from metapub import PubMedFetcher 
import os
os.environ['NCBI_API_KEY'] = NCBI_API_KEY
#For label studio
import requests
from label_studio_sdk import Client
#For training
import json
import spacy
from spacy.training import Example, offsets_to_biluo_tags
from spacy.training.iob_utils import biluo_tags_to_offsets
from spacy.training.example import Example

#Fetch records from PubMed
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
# See if we can fine-tune an NER to pull useful info from abstracts
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

# Loop through each PDF in the directory
for a in articles:
	# Upload abstracts
	abstract = a.abstract
	if abstract: # Check if the article has an abstract
		upload_task(abstract, PROJECT_ID)
	else:
		print(f"No abstract found for article with PMID: {article.pmid}")


#==============================================================================
#NER training portion -- This section is for the first time that you train a 
#model. See below for the updates to an existing model. 
#==============================================================================
latest_labels = 'project-2-at-2024-06-12-18-41-71e3df69.json'

latest_labels = 'project-2-at-2024-06-12-18-46-71e3df69.json'
# Load the exported data from Label Studio
with open(latest_labels, 'r', encoding='utf-8') as file:
    labeled_data = json.load(file)

# Function to filter out overlapping entities
def filter_overlapping_entities(entities):
    entities = sorted(entities, key=lambda x: (x[0], x[1]))  # Sort by start and end
    filtered_entities = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            filtered_entities.append((start, end, label))
            last_end = end
    return filtered_entities

#Some entries return warnings that they are misaligned. Use
#this function to fix that. 
def fix_misaligned_entities(train_data, nlp_use):
    fixed_train_data = []
    for text, annotations in train_data:
        doc = nlp_use.make_doc(text)
        entities = annotations.get("entities")
        # Check alignment
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        if '-' in biluo_tags:
            # Fix misaligned entities
            corrected_entities = biluo_tags_to_offsets(doc, biluo_tags)
            annotations["entities"] = corrected_entities
        fixed_train_data.append((text, annotations))
    return fixed_train_data

# Convert labeled data to SpaCy format
TRAIN_DATA = []
for item in labeled_data:
    text = item['data'].get('text', '')  # Use .get to avoid KeyError if 'text' is missing
    entities = []
    for annotation in item.get('annotations', []):
        for result in annotation.get('result', []):
            value = result.get('value', {})
            start = value.get('start', 0)
            end = value.get('end', 0)
            labels = value.get('labels', [])
            if labels:
                label = labels[0]
                entities.append((start, end, label))
    filtered_entities = filter_overlapping_entities(entities)
    TRAIN_DATA.append((text, {"entities": filtered_entities}))

print(TRAIN_DATA)

# Load the pre-trained SciSpacy model
nlp = spacy.load("en_core_sci_md")

#Fix the training data.
fixed_train_data = fix_misaligned_entities(TRAIN_DATA,nlp)

# Get the NER component
ner = nlp.get_pipe("ner")

# Add custom entity labels
labels = set()
for _, annotations in TRAIN_DATA:
    for ent in annotations['entities']:
        labels.add(ent[2])

for label in labels:
    ner.add_label(label)

# Prepare training data
examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Disable other pipelines during training
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Start the training
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.resume_training()
    for i in range(40):  # Number of iterations
        print(f"Iteration {i+1}")
        losses = {}
        nlp.update(
            examples,
            drop=0.35,  # Dropout - make it harder to memorize data
            losses=losses,
        )
        print(losses)

# Save the model
output_dir = "custom_sci_ner_abs"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Load the trained model
output_dir = "custom_sci_ner_abs"
nlp_cust = spacy.load(output_dir)

# Test the model by fetching a task from label studio
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

# Get the project
project = ls.get_project(PROJECT_ID)

# Fetch tasks from the project
tasks = project.get_tasks()

#Pick one to try
test_text = tasks[20]['data']['text']

doc = nlp_cust(test_text)

# Extract entities and their labels
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Extracted Entities:")
for entity in entities:
    print(entity)

#==============================================================================
#Link to Label Studio and generate suggested labels. Turn this into a loop to 
#get better predictions as more annotations are added. 
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

#==============================================================================
#NER training portion -- This section is for repeated trainings of the same 
# model on updated labels. 
#==============================================================================
latest_labels = 'project-2-at-2024-06-12-13-51-b02f25ff.json'

# Load the exported data from Label Studio
with open(latest_labels, 'r', encoding='utf-8') as file:
    labeled_data = json.load(file)

#Filter out entries that have not been annotated by a human: 
labeled_data = [task for task in labeled_data if 'annotations' in task and task['annotations']]

# Function to filter out overlapping entities
def filter_overlapping_entities(entities):
    entities = sorted(entities, key=lambda x: (x[0], x[1]))  # Sort by start and end
    filtered_entities = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            filtered_entities.append((start, end, label))
            last_end = end
    return filtered_entities

#Some entries return warnings that they are misaligned. Use
#this function to fix that. 
def fix_misaligned_entities(train_data, nlp):
    fixed_train_data = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        entities = annotations.get("entities")
        # Check alignment
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        if '-' in biluo_tags:
            # Fix misaligned entities
            corrected_entities = biluo_tags_to_offsets(doc, biluo_tags)
            annotations["entities"] = corrected_entities
        fixed_train_data.append((text, annotations))
    return fixed_train_data


# Convert labeled data to SpaCy format
TRAIN_DATA = []
for item in labeled_data:
    text = item['data'].get('text', '')  # Use .get to avoid KeyError if 'text' is missing
    entities = []
    for annotation in item.get('annotations', []):
        for result in annotation.get('result', []):
            value = result.get('value', {})
            start = value.get('start', 0)
            end = value.get('end', 0)
            labels = value.get('labels', [])
            if labels:
                label = labels[0]
                entities.append((start, end, label))
    filtered_entities = filter_overlapping_entities(entities)
    TRAIN_DATA.append((text, {"entities": filtered_entities}))

print(TRAIN_DATA)

# Load the trained model
output_dir = "custom_sci_ner_abs"
nlp_cust = spacy.load(output_dir)

#Fix the training data.
fixed_train_data = fix_misaligned_entities(TRAIN_DATA,nlp_cust)

# Get the NER component
ner = nlp_cust.get_pipe("ner")

# Add custom entity labels
labels = set()
for _, annotations in fixed_train_data:
    for ent in annotations['entities']:
        labels.add(ent[2])

for label in labels:
    ner.add_label(label)

# Prepare training data
examples = []
for text, annotations in fixed_train_data:
    doc = nlp_cust.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Disable other pipelines during training
unaffected_pipes = [pipe for pipe in nlp_cust.pipe_names if pipe != "ner"]

# Start the training
with nlp_cust.disable_pipes(*unaffected_pipes):
    optimizer = nlp_cust.resume_training()
    for i in range(40):  # Number of iterations
        print(f"Iteration {i+1}")
        losses = {}
        nlp_cust.update(
            examples,
            drop=0.35,  # Dropout - make it harder to memorize data
            losses=losses,
        )
        print(losses)

# Save the model
output_dir = "custom_sci_ner_abs"
nlp_cust.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Test the model by fetching a task from label studio
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

# Get the project
project = ls.get_project(PROJECT_ID)

# Fetch tasks from the project
tasks = project.get_tasks()

#Pick one to try
test_text = tasks[40]['data']['text']

doc = nlp_cust(test_text)

# Extract entities and their labels
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Extracted Entities:")
for entity in entities:
    print(entity)