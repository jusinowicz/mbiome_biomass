
# Function to parse entities and organize data
def parse_entities(entities):
	treatment = {}
	response = {}
	for entity, label, start, end in entities:
		if label in ["TREATMENT", "INOCTYPE", "FIELDGREENHOUSE", "SOILTYPE", "ECOTYPE", "ECOREGION", "LANDUSE", "LOCATION"]:
			treatment[label] = entity
		elif label in ["RESPONSE"]:
			response[label] = entity
		elif label in ["CARDINAL", "PERCENTAGE", "UNITS"]:
			response[label] = entity
	# Assuming that treatment and response can be combined based on some logic
	treatment_key = tuple(treatment.items())
	data[treatment_key].append(response)

# Enhanced function to parse context-dependent relationships
def parse_entities(entities):
	treatments = []
	responses = []
	percentages = []
	cardinal = None
	for entity, label, start, end in entities:
		if label == "TREATMENT":
			treatments.append(entity)
		elif label == "RESPONSE":
			responses.append(entity)
		elif label == "PERCENTAGE":
			percentages.append(entity)
		elif label == "CARDINAL":
			cardinal = entity
	# Ensure we have consistent lengths for treatments and percentages
	# Here, treatments are expected to appear in groups with associated percentages
	data = []
	for treatment in treatments:
		for percentage in percentages:
			data.append({
				"TREATMENT": treatment,
				"RESPONSE": responses[0] if responses else None,
				"CARDINAL": cardinal if cardinal else None,
				"PERCENTAGE": percentage
			})
	return data

if ent[1] == 'TREATMENT' or ent[1] == 'INOCTYPE':

# Enhanced function to parse context-dependent relationships
def parse_entities(entities):
	# Temporary storage for the entities
	treatments = []
	responses = []
	percentages = []
	cardinal_numbers = []
	# Extract entities and organize them by type
	for entity, label, start, end in entities:
		if label in ["TREATMENT", "INOCTYPE"]:
			treatments.append((entity, start))
		elif label == "RESPONSE":
			responses.append((entity, start))
		elif label == "CARDINAL":
			cardinal_numbers.append((entity, start))
		elif label == "PERCENTAGE":
			percentages.append((entity, start))
	# Ensure responses and cardinal are properly recorded
	response = responses[0] if responses else None
	# Group treatments by proximity
	treatment_groups = defaultdict(list)
	for treatment, position in treatments:
		treatment_groups[position].append(treatment)
	# Group percentages and cardinal numbers with treatments by context
	treatment_combinations = []
	previous_treatment = None
	for treatment_position in sorted(treatment_groups.keys()):
		current_treatment = treatment_groups[treatment_position]
		if previous_treatment is not None:
			for pct, position in percentages:
				if previous_treatment[1] < position < treatment_position:
					treatment_combinations.append((previous_treatment[0], current_treatment[0], pct))
		previous_treatment = (current_treatment, treatment_position)
	data = []
	percentage_index = 0
	for base_treatment, compare_treatment, pct in treatment_combinations:
		if percentage_index < len(percentages):
			data.append({
				"TREATMENT": f"{base_treatment} and {compare_treatment}",
				"RESPONSE": response,
				"CARDINAL": cardinal_numbers[percentage_index][0] if percentage_index < len(cardinal_numbers) else "NA",
				"PERCENTAGE": pct
			})
			percentage_index += 1
		return data


def filter_relevant_entities(entities):
	treatments = []
	responses = []
	cards = []
	percentages = []
	for ent in entities:
		if ent[1] == 'TREATMENT' or ent[1] == 'INOCTYPE':
			treatments.append(ent)
		elif ent[1] == 'RESPONSE' and ('biomass' in ent[0].lower() or 'dry weight' in ent[0].lower()):
			responses.append(ent)
		elif ent[1] == 'CARDINAL':
			cards.append(ent)
		elif ent[1] == 'PERCENTAGE':
			percentages.append(ent)
	return treatments, responses, cards, percentages

def extract_relationships(doc, treatments, responses):
	treatment_response_pairs = []
	for treatment in treatments:
		for response in responses:
			# Check if they belong to the same sentence
			if treatment[2] // len(doc) == response[2] // len(doc):
				treatment_response_pairs.append((treatment, response))
			else:
				# Use dependency parsing to find if there's a direct relationship
				treatment_span = doc[treatment[2]:treatment[3]]
				response_span = doc[response[2]:response[3]]
				if treatment_span.root.head == response_span.root or response_span.root.head == treatment_span.root:
					treatment_response_pairs.append((treatment, response))
	return treatment_response_pairs

def build_table(treatment_response_pairs, cards, percentages):
	data = []
	for i, (treatment, response) in enumerate(treatment_response_pairs):
		card = cards[i][0] if i < len(cards) else 'NA'
		percentage = percentages[i][0] if i < len(percentages) else 'NA'
		data.append([treatment[0], response[0], card, percentage])
	df = pd.DataFrame(data, columns=['TREATMENT', 'RESPONSE', 'CARDINAL', 'PERCENTAGE'])
	return df

# Example usage
text = "The plant dry weight was improved with the application of Bradyrhizobium by 59.3, 13.5 and 34.8%; and with the application of AMF by 63.2, 100% NPK and 50% NPK, respectively"
doc, entities = extract_entities(text)
treatments, responses, cards, percentages = filter_relevant_entities(entities)
treatment_response_pairs = extract_relationships(doc, treatments, responses)
table = build_table(treatment_response_pairs, cards, percentages)

print(table)

##############

def extract_entities_and_dependencies(sentences):
	all_entities = []
	docs = []
	for sent in sentences:
		doc = nlp(sent)
		docs.append(doc)
		entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
		all_entities.append((doc, entities))
	return docs, all_entities

# Function to filter relevant entities
def filter_relevant_entities(entities):
	treatments = []
	responses = []
	cards = []
	percentages = []
	for ent in entities:
		if ent[1] == 'TREATMENT':
			treatments.append(ent)
		elif ent[1] == 'RESPONSE' and ('biomass' in ent[0].lower() or 'dry weight' in ent[0].lower() or 'yield' in ent[0].lower()):
			responses.append(ent)
		elif ent[1] == 'CARDINAL':
			cards.append(ent)
		elif ent[1] == 'PERCENTAGE':
			percentages.append(ent)
	return treatments, responses, cards, percentages

# Function to extract relationships between treatments and responses
def extract_relationships(docs, entities):
	treatment_response_pairs = []
	for doc, ents in zip(docs, entities):
		treatments, responses, _, _ = filter_relevant_entities(ents)
		for treatment in treatments:
			for response in responses:
				# Check if they belong to the same sentence
				treatment_span = doc[treatment[2]:treatment[3]]
				response_span = doc[response[2]:response[3]]
				if treatment_span.sent == response_span.sent:
					treatment_response_pairs.append((treatment, response))
				else:
					# Use dependency parsing to find if there's a direct relationship
					if treatment_span.root.head == response_span.root or response_span.root.head == treatment_span.root:
						treatment_response_pairs.append((treatment, response))
	return treatment_response_pairs

# Function to build the table from extracted relationships
def build_table(treatment_response_pairs, cards, percentages):
	data = []
	for i, (treatment, response) in enumerate(treatment_response_pairs):
		card = cards[i][0] if i < len(cards) else 'NA'
		percentage = percentages[i][0] if i < len(percentages) else 'NA'
		data.append([treatment[0], response[0], card, percentage])
	df = pd.DataFrame(data, columns=['TREATMENT', 'RESPONSE', 'CARDINAL', 'PERCENTAGE'])
	return df
############


# Function to extract relationships between treatments and responses
def extract_relationships(docs, entities):
	treatment_response_pairs = []
	for doc, ents in zip(docs, entities):
		treatments, responses, _, _ = filter_relevant_entities(ents)
		for treatment in treatments:
			for response in responses:
				# Check if they belong to the same sentence
				if treatment[2] // len(doc) == response[2] // len(doc):
					treatment_response_pairs.append((treatment, response))
				else:
					# Use dependency parsing to find if there's a direct relationship
					treatment_span = doc[treatment[2]:treatment[3]]
					response_span = doc[response[2]:response[3]]
					if treatment_span.root.head == response_span.root or response_span.root.head == treatment_span.root:
						treatment_response_pairs.append((treatment, response))
	return treatment_response_pairs

def build_table(treatment_response_pairs, cards, percentages):
	data = []
	for i, (treatment, response) in enumerate(treatment_response_pairs):
		card = cards[i][0] if i < len(cards) else 'NA'
		percentage = percentages[i][0] if i < len(percentages) else 'NA'
		data.append([treatment[0], response[0], card, percentage])
	df = pd.DataFrame(data, columns=['TREATMENT', 'RESPONSE', 'CARDINAL', 'PERCENTAGE'])
	return df

	def create_table(doc, entities, treatments):
	data = []
	responses = ['dry weight', 'biomass']
	for treatment in treatments:
		print(f"Treatment: {treatment}")
		for response in responses:
			response_ents = [ent for ent in entities if ent[1] == 'RESPONSE' and response in ent[0].lower()]
			print(f"Response: {response}")
			for resp_ent in response_ents:
				resp_span = doc[resp_ent[2]:resp_ent[3]]
				for token in resp_span.root.head.subtree:
 					# Extract numerical values and percentage signs while avoiding punctuation
					if token.ent_type_ in ['CARDINAL', 'PERCENTAGE'] and (token.is_digit or token.like_num or token.text == '%'):
						value = token.text
						if value == '%':
							if data and data[-1]['PERCENTAGE'] == '':
								data[-1]['PERCENTAGE'] = value
							else:
								print(f"Response subtree, True: {treatment} {response} {value}")
								data.append({
									'TREATMENT': treatment,
									'RESPONSE': response,
									'CARDINAL': '',
									'PERCENTAGE': value
								})
						else:
							data.append({
								'TREATMENT': treatment,
								'RESPONSE': response,
								'CARDINAL': value,
								'PERCENTAGE': ''
							})
	# Remove duplicate percentage entries
	clean_data = []
	for entry in data:
		if entry['PERCENTAGE'] == '%':
			continue
		clean_data.append(entry)
	df = pd.DataFrame(data)
	return df



# Function to identify treatments from entities
def identify_treatments(entities):
	treatments = set()
	for entity in entities:
		if entity[1] == 'TREATMENT' or entity[1] == 'INOCTYPE':
			treatments.add(entity[0])
	return list(treatments)

# Function to create a table of treatments and responses
def create_table(doc, entities, treatments):
	data = []
	responses = ['dry weight', 'biomass']
	for treatment in treatments:
		print(f"Treatment: {treatment}")
		for response in responses:
			response_ents = [ent for ent in entities if ent[1] == 'RESPONSE' and response in ent[0].lower()]
			print(f"Response: {response}")
			for resp_ent in response_ents:
				resp_span = doc[resp_ent[2]:resp_ent[3]]
				for token in resp_span.root.head.subtree:
					if token.ent_type_ in ['CARDINAL', 'PERCENTAGE']:
						value = token.text
						treatment_related = False
						for tok in resp_span.root.head.subtree:
							print(f"Response subtree, : {tok}")
							#if tok.text == treatment:
							#Try this version for more robust matching:
							if bool(set(tok.text.split()) & set(treatment.split())):
								treatment_related = True
								print(f"Response subtree, False, but: {tok.text}")
								break
						if treatment_related:
							print(f"Response subtree, True: {treatment} {response} {value}")
							data.append({
								'TREATMENT': treatment,
								'RESPONSE': response,
								'CARDINAL': '',
								'PERCENTAGE': value
							})
	df = pd.DataFrame(data)
	return df


def extract_values(token, treatment, response):
	values = []
	for child in token.children:
		if child.ent_type_ in ['CARDINAL', 'PERCENTAGE'] and (child.is_digit or child.like_num or '%' in child.text):
			value = child.text.strip(',')
			values.append((value, '%' if '%' in value else ''))
		elif child.dep_ in ['conj', 'appos'] and child.head == token:
			values.extend(extract_values(child, treatment, response))
	return values


def create_table(doc, entities):
	data = []
	responses = ['dry weight', 'biomass', 'plant growth']
	treatment_entities = [ent for ent in entities if ent[1] == 'TREATMENT' or ent[1] == 'INOCTYPE']
	for treatment_entity in treatment_entities:
		treatment = treatment_entity[0]
		for response in responses:
			response_ents = [ent for ent in entities if ent[1] == 'RESPONSE' and response in ent[0].lower()]
			for resp_ent in response_ents:
				resp_span = doc[resp_ent[2]:resp_ent[3]]
				head_token = resp_span.root.head
				for token in head_token.children:
					if token.ent_type_ in ['CARDINAL', 'PERCENTAGE'] or token.dep_ in ['nmod', 'appos', 'conj']:
						values = extract_values(token, treatment, response)
						for value, percent in values:
							data.append({
								'TREATMENT': treatment,
								'RESPONSE': response,
								'CARDINAL': value,
								'PERCENTAGE': percent
							})
	df = pd.DataFrame(data)
	return df

#################################
#Step 5. Run NER to extract entities
#Load the model
nlp = spacy.load(output_dir)
# def extract_entities(text):
# 	doc = nlp(text)
# 	#This line is for extracting entities with dependencies. 
# 	entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
# 	return doc,entities


def filter_sentences(doc, keywords):
	"""
	Filter sentences in the parsed doc that contain any of the specified keywords.

    Args:
        doc: The parsed spaCy Doc object.
        keywords: A list of keywords to filter sentences by.

    Returns:
        A new spaCy Doc object containing only the sentences with keywords.
    """
	keywords = [keyword.lower() for keyword in keywords]

	filtered_spans = []

	for sent in doc.sents:
		# Check if any of the keywords are in the sentence
		if any(keyword in sent.text.lower() for keyword in keywords):
			filtered_spans.append(sent)

	# Create a new doc with only the filtered sentences
	filtered_tokens = [token for span in filtered_spans for token in span]

	# Create the new Doc object with the filtered tokens
	filtered_doc = Doc(doc.vocab, words=[token.text for token in filtered_tokens])

	# Copy annotations from the original doc to the new doc
	for name, proc in nlp.pipeline:
		filtered_doc = proc(filtered_doc)
	
	return filtered_doc


#Function to trace syntactical dependency back to a specific label
#Use this to find the TREATMENT corresponding to a CARDINAL or PERCENTAGE
def find_label_in_tree(token, label_id):
	vnames = []
	for ancestor in token.ancestors:
		for child in ancestor.children:
			if child.ent_type_ in label_id:
				vname = child.text.strip(',')
				vnames.append(vname)
			elif child.dep_ in ['nmod','nummod','conj', 'appos']:
				find_label_in_tree(ancestor, label_id)
	return vnames

# Function to create a table of treatments and responses
def create_table(doc, entities):
	data = []
	responses = ['dry weight', 'biomass']
	for response in responses:
		response_ents = [ent for ent in entities if ent[1] == 'RESPONSE' and response in ent[0].lower()]
		for resp_ent in response_ents:
			resp_span = doc[resp_ent[2]:resp_ent[3]]
			for token in resp_span.root.head.subtree:
				#Check it's a type we want, and not punctuation
				if token.ent_type_ in ['CARDINAL', 'PERCENTAGE'] and token.text not in ['%', ' ', ',']:
					value = token.text
					#Find the connected treatment by parsing dependencies
					treat = find_label_in_tree(token, label_id)
					if token.ent_type_ == 'CARDINAL':
						data.append({
							'TREATMENT': treat,
							'RESPONSE': response,
							'CARDINAL': value,
							'PERCENTAGE': ''
						})
					elif token.ent_type_ == 'PERCENTAGE':
						data.append({
							'TREATMENT': treat,
							'RESPONSE': response,
							'CARDINAL': '',
							'PERCENTAGE': value
						})
	df = pd.DataFrame(data)
	return df


results_text = sections["results"]
results_text = " ".join(results_text)
results_doc = nlp(results_text)
#This line is for extracting entities with dependencies. 
results_entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in results_doc.ents]

#Filter sentences in the "Results" section
keywords = ["biomass", "dry weight", "yield"]
results_doc_filtered = filter_sentences(results_doc, keywords) 
results_entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in results_doc_filtered.ents]
table = create_table(results_doc_filtered, results_entities)


############################

