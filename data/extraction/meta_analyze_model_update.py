#==============================================================================
# This is a third key chunk of code that is responsible for loading in the 
# annotated data from Label Studio, cleans, converts it, and then fits the 
# NER. 
# Fitting the NER can either be done from scratch, or by loading the custom NER
# and training it on new labels. 
#==============================================================================
#==============================================================================
# Libraries
#==============================================================================
import json 
import spacy
from spacy.training.example import Example
#==============================================================================
#==============================================================================
#Load and clean data
#==============================================================================
latest_labels = 'project-2-at-2024-06-13-10-25-2c35c7e5.json'

# Load the exported data from Label Studio
with open(latest_labels, 'r', encoding='utf-8') as file:
    labeled_data = json.load(file)

#Step 1: Filter out entries that have not been annotated by a human: 
labeled_data = [task for task in labeled_data if 'annotations' in task and task['annotations']]

#Step 2: Clean Annotations Function
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

cleaned_data = clean_annotations(labeled_data)

#==============================================================================
#Load and fit the model
#==============================================================================

#LOAD nlp the FIRST time or to retrain from scratch
#Load the spaCy model
nlp = spacy.load("en_core_sci_md")

#OR retrain a model on new data
# output_dir = "custom_sci_ner_abs"
# nlp = spacy.load(output_dir)

# Prepare the data for spaCy
examples = []
for text, annotations in cleaned_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Add the new labels to the NER component
ner = nlp.get_pipe("ner")
labels = set(label for _, anns in cleaned_data for _, _, label in anns["entities"])
for label in labels:
    ner.add_label(label)

# Disable other pipes for training
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.resume_training()
    for i in range(150):  # Number of iterations
        print(f"Iteration {i+1}")
        losses = {}
        nlp.update(
            examples,
            drop=0.35,  # Dropout - make it harder to memorize data
            losses=losses,
        )
        print(losses) 

# Save the model
output_dir = "custom_sci_ner_abs_v2"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
