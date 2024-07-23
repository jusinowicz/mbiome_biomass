#==============================================================================
# For the meta analysis and database: 
# This is STEP 4 in the pipeline:
# 
# Use NLP and a custom NER model to extract the TREATMENTs and RESPONSEs from 
# the text of the Methods and Results sections in scientfic papers. 
#
# This is meant to be the 1st step in paper parsing, trying to glean info from 
# the text before trying more complex table-extraction and figure-extraction
# methods. 
# 
# Current NER: custom_web_ner_abs_v382
#
# The final output is a df/CSV with columns STUDY, TREATMENT, RESPONSE, CARDINAL,
# PERCENTAGE, SENTENCE, ISTABLE. 
# There are separate columns for CARDINAL (a numeric)
# response) and PERCENTAGE because the NER recognizes them separately. This is 
# useful because it helps determine whether actual units of biomass response are 
# being identified or the ratio of response (percentage). 
#
# SENTENCE is the sentence that was parsed for the information in the table 
# ISTABLE is a numeric ranking from 0-2 which suggests how likely it is that the
# the parsed information came from a table that the pdf-parsing grabbed. In this
# case, the results are most definitely not to be trusted. 
#
# This table is meant to help determine what information is available in the paper
# and indicate whether further downstream extraction is necessary. 
#
# See extrat_abstract_dat for installation notes.  
#
# This code works fairly well now, but further downstream processing coul be 
# implemented to help human eyes interpret and sift through the useful information.
# In partiular, removing (or at least flagging) entries that appear to be numbers 
# grabbed from summary statistics (e.g. p-values, F-values, AIC, etc.). This seems 
# to happen frequently. 
#==============================================================================
#py -3.8

#Libraries
import pandas as pd

#NER and NLP
import spacy


#Post-NER processing
from collections import defaultdict

#The shared custom definitions
#NOTE: This line might have to be modified as structure changes and 
#we move towards deployment
## Add the project root directory to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(os.path.abspath('./../'))  
import common.utilities

#==============================================================================
#Set paths: 
#==============================================================================
#Current NER to use: 
output_dir = "./../models/custom_web_ner_abs_v382"
#Load the model
nlp = spacy.load(output_dir)

#Where the papers live 
pdf_dir = "./../papers/" 
#==============================================================================
# This section loads the PDFs from a folder and gets the right sections: 
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

#Step 4. Filter sentences containing specific keywords
#Step 5. Run NER to extract entities
#Step 6. Use the dependency parsing of the sentences done by spacy NLPs to 
#infer relationships between RESPONSE, TREATMENT, and CARDINAL/PERCENTAGE to
#ultimately build out a table of these relationships. 
#For debug: pdf_path = "./papers/33888066.pdf"
#pdf_path = "./papers/33893547.pdf"
#pdf_path = "./papers/35410135.pdf"

# Get the list of current PDFs in the directory

new_pdfs = {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

# Process the PDFs
data = []
for pdf in new_pdfs:
	#1.
	pdf_path = pdf_dir + pdf
	study_id = pdf.rstrip(".pdf")
	pdf_text = extract_text_from_pdf(pdf_path)
	#2. 
	sentences = preprocess_text(pdf_text)
	#3.
	sections = identify_sections(sentences)
	#4.
	# #Get the methods
	# methods_text = " ".join(sections.get('methods', []))
	# methods_doc, methods_entities = extract_entities(methods_text)
	# data_methods=parse_entities(methods_entities) 
	#Filter sentences in the "Results" section
	keywords = ["biomass", "dry weight", "yield"]
	results_text = filter_sentences(sections["results"], keywords) 
	#Extract entities from filtered text
	results_text = " ".join(results_text)
	# Remove remaining newline characters
	results_text = re.sub(r'\n', ' ', results_text)
	results_doc, results_entities = extract_entities(results_text)
	table = create_table(results_doc, results_entities, study_id)
	data.append(table)


flattened_data = [item for sublist in data for item in sublist]
df = pd.DataFrame(flattened_data)

# Export DataFrame to a CSV file
df.to_csv('./output/extract_from_text2.csv', index=False)

# Export DataFrame to a CSV file
new_df = df[df["ISTABLE"] == 0] 
new_df.to_csv('extract_correct_text1.csv', index=False)


#==============================================================================
# This section contains miscellaneous tools for parsing and visualizing the 
# sentence dependency structures. This might be temporary
#==============================================================================
# Generate the dependency tree in a textual format
# Access the specific sentence using doc.sents
sentence_index = 6
sentence = list(results_doc.sents)[sentence_index]

# Function to recursively print the tree structure
def print_tree(token, indent=''):
	print(f"{indent}{token.text} <-- {token.dep_} <-- {token.head.text}")
	for child in token.children:
		print_tree(child, indent + '  ')

for token in sentence:
    print_tree(token)

# Save the syntacticdependency visualization to an HTML file
from spacy import displacy
html = displacy.render(sentence, style="dep", page=True)
with open("./output/syntactic_tree_ex4.html", "w", encoding="utf-8") as file:
    file.write(html)

# Generate the dependency tree in html
# displacy.render(doc, style="dep", options={"compact": True, "color": "blue"})
# tree = displacy.render(sentence, style="dep", options={"compact": True, "color": "blue"})


s1 = "The plant dry weight was improved with the application of Bradyrhizobium by 59.3, 13.5, and 34.8%; and with the application of AMF by 63.2, 21.8, and 41.0% and with their combination by 61.7, 18.7, 38.7% in both growing seasons as compare with control, 100% NPK and 50% NPK respectively(Table 2)."
s2 = "The applications of fertilizer and AMF increased the dry weight by 100 and 300%, respecticely."
s3 = "The application of fertilizer increased the dry weight by 100%, while the application of AMF increased the dry weight by 300%."
s4 = "The highest dry biomass shoot found was 10.39 g  and root 9.59 g/plant in T3 inoculated with AMF in  T. arjuna, which was 29.71% and 19.72% higher  compared to non-inoculated control plants grown in  the same ratio of soil (Table 2)."



