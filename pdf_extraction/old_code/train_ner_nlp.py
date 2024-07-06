#==============================================================================
# Use the labeled data set from Label Studio to train and fine tune
# the custom NER based on 
# py -3.10 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
#
#==============================================================================
py -3.10 

# with open('project-1-at-2024-06-10-23-20-91f8e0f3.json', 'rb') as file:
#     raw_data = file.read()
#     result = chardet.detect(raw_data)
#     encoding = result['encoding']
#     print(f"Detected encoding: {encoding}")

import json
import spacy
from spacy.training.example import Example

# Load the exported data from Label Studio
with open('project-1-at-2024-06-10-23-20-91f8e0f3.json', 'r', encoding='utf-8') as file:
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
    for i in range(30):  # Number of iterations
        print(f"Iteration {i+1}")
        losses = {}
        nlp.update(
            examples,
            drop=0.35,  # Dropout - make it harder to memorize data
            losses=losses,
        )
        print(losses)

# Save the model
output_dir = "custom_sci_ner_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Load the trained model
nlp_cust = spacy.load(output_dir)

# Test the model
test_text = "Plant growth in inoculated soil was significantly higher compared to non-inoculated soil."
doc = nlp_cust(test_text)

# Extract entities and their labels
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Extracted Entities:")
for entity in entities:
    print(entity)
