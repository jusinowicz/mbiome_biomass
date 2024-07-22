#==============================================================================
# All of the custom functions that are used across modules are collected here
# 
#==============================================================================
#Libraries
#==============================================================================
import requests #upload_task

#==============================================================================
# Functions for dealing with label studio 
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
# Functions related to spacy  
#==============================================================================
#Clean Annotations Function
#This handy function converts the JSON format to the correct format
#for spacy, deals with misaligned spans, and removes white space and 
#punctuation in the spans. 

def clean_annotations(data):
    cleaned_data = []
    for item in data:
        text = item['data']['text']
        entities = []
        for annotation in item['annotations']:
            for result in annotation['result']:
                value = result['value']
                start, end, label = value['start'], value['end'], value['labels'][0]
                entity_text = text[start:end]
                # Remove leading/trailing whitespace from entity spans
                while entity_text and entity_text[0].isspace():
                    start += 1
                    entity_text = text[start:end]
                while entity_text and entity_text[-1].isspace():
                    end -= 1
                    entity_text = text[start:end]
                # Check for misaligned entries and skip if misaligned
                if entity_text == text[start:end]:
                    entities.append((start, end, label))
        if entities:
            cleaned_data.append((text, {"entities": entities}))
    return cleaned_data


#This function will return the text and the entities for processing
def extract_entities(text):
	doc = nlp(text)
	#This line is for extracting entities only
	#entities = [(ent.text, ent.label_) for ent in doc.ents]
	#This line is for extracting entities with dependencies. 
	entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
	return doc,entities


#This function is to group and refine within each entity group
def find_entity_groups(doc, entities, label_type):
	# Create a dictionary mapping token indices to entities of the given label type
	entity_dict = {ent[2]: (ent[0], ent[1]) for ent in entities if ent[1] == label_type}
	entity_groups = []
	for sent in doc.sents:
		sent_entities = {token.i: entity_dict[token.i] for token in sent if token.i in entity_dict}
		sent_entity_groups = []
		for token in sent:
			if token.i in sent_entities and sent_entities[token.i][1] == label_type:
				entity_group = [sent_entities[token.i][0]]
				# Find modifiers or compound parts of the entity using dependency parsing
				for child in token.children:
					if child.dep_ in ['amod', 'compound', 'appos', 'conj', 'advmod', 'acl', 'prep', 'pobj', 'det']:
						if child.i in sent_entities:
							entity_group.append(sent_entities[child.i][0])
						else:
							entity_group.append(child.text)
				# Also check if the token itself has a head that is an entity of the same type
				if token.head.i in sent_entities and sent_entities[token.head.i][1] == label_type and token.head != token:
					entity_group.append(token.head.text)
				# Sort and join entity parts to maintain a consistent order
				entity_group = sorted(entity_group, key=lambda x: doc.text.find(x))
				sent_entity_groups.append(" ".join(entity_group))
		if sent_entity_groups:
			entity_groups.extend(sent_entity_groups)
	# Removing duplicates and returning the result
	return list(set(entity_groups))

#==============================================================================
# Functions for dealing with label studio 
#==============================================================================
