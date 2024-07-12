#https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb
#py -3.8 -m pip install torchvision timm

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML
from itertools import zip_longest
import pandas as pd
import numpy as np 
import deepdoctection as dd
import re

#path = Path.cwd() / "pics/samples/sample_2"
path = "./../papers/21222096.pdf"
#path = "./../papers/27287440.pdf"

analyzer = dd.get_dd_analyzer()
df = analyzer.analyze(path=path)
df.reset_state()  # This method must be called just before starting the iteration. It is part of the API.

pages =[]
for doc in df: 
    pages.append(doc)


table = pages[6].tables[0]
t1 = table.csv
#t2 = pd.DataFrame(t1[1:],columns = [0,1])
t2 = pd.DataFrame(t1)

#Replace blank space with NaN
t2.replace(r'^\s*$', np.nan, regex=True, inplace=True)

#Drop columns and rows of NaN created by spaces
t2.dropna(axis=1, how='all', inplace=True)
t2.dropna(axis=0, how='all', inplace=True)

# Remove letters and special symbols from numbers
# Remove leading/trailing whitespace from all cells and make all 
# lowercase
def clean_numbers(cell):
    if isinstance(cell, str):
        # Remove leading and trailing whitespace
        cell = cell.strip().lower()
        # Remove special characters
        cell = re.sub(r'[^\w\s]', '', cell)
        # Remove non-numeric characters after numbers
        cell = re.sub(r'(\d+(\.\d+)?)(\s*[a-zA-Z]+)?', r'\1', cell)
        # Convert to float if numeric
        if not pd.isna(cell):
            try:
                cell = float(cell)
            except ValueError:
                pass  # If conversion fails, cell remains unchanged    
    return cell

t2= t2.applymap(clean_numbers)

#Fill NaN with previous cell in row
t2 = t2.fillna(method='ffill', axis = 1)  

#Try to detect and combine header rows
def classify_cells(df):
    classifications = pd.DataFrame(index=df.index, columns=df.columns)
    for row in df.index:
        for col in df.columns:
            cell = df.at[row, col]
            if pd.isna(cell):
                classifications.at[row, col] = 'NaN'
            elif isinstance(cell, str):
                classifications.at[row, col] = 'string'
            elif isinstance(cell, (int, float)):
                classifications.at[row, col] = 'numeric'
            else:
                classifications.at[row, col] = 'other'
    return classifications

# Apply the classification function to each cell in the DataFrame
classified_t2 =classify_cells(t2)  

# Function to check if cell type is same as previous cell in the column
def is_same_type(df):
    results = pd.DataFrame(index=df.index, columns=df.columns, dtype=bool)
    for col in df.columns:
        for row in df.index[1:]:  # Start from the second row
            current_type = (df.at[row, col])
            previous_type = (df.at[row-1, col])
            results.at[row, col] = current_type == previous_type
    return results

same_type = is_same_type(classified_t2)

#Use this information to find column headers and parse the table
#Try to infer which rows are likely to contain headers based on 
#where the data type across a row changes. If there seem to be 
#multiple header rows then divide the table into multiple tables. 

def organize_tables(table, same_type):
    final_tables = []
    #Determine which case we're dealing with:
    #Number of (sub)header rows:
    nheaders = len(same_type)-same_type.sum()
    nheaders = nheaders.iloc[1]
    #Case 0: First row is headers, single row
    if nheaders == 0:
        final_tables = pd.DataFrame(table.iloc[1:,:])
        final_tables = final_tables.rename(columns = (table.iloc[0,:]))
    #Case 1: Multiple headers at the top of table
    elif nheaders == 1:
        # Find the index where the type switch occurs (row with False values)
        # Ignore the first row and first column, then find the first occurrence of False
        header_index = same_type.iloc[1:, 1:].eq(False).idxmax().max() 
        # Concatenate rows 0 to header_index to create new headers
        header_rows = table.iloc[:header_index]
        new_headers = header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0)
        # Create the new DataFrame with the remaining rows and the new headers
        data_rows = table.iloc[header_index + 1:]
        final_tables = pd.DataFrame(data_rows.values, columns=new_headers)
    #Case 2: Multiple headers and sub-headers within table 
    elif nheaders > 1:
        #Assume that the first block of headers includes both the main overall 
        #headers, as well as the first row of subheaders: 
        header_index = same_type.iloc[1:, 1:].eq(False).idxmax().max()
        # Concatenate rows 0 to header_index-1 to exclude first sub-header and
        # create new main headers
        header_rows = table.iloc[:header_index-1]
        new_headers = header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0) 
        # Iterate through the table to find additional sub-headers and divide the table into sub-sections
        current_index = header_index-1
        while current_index < len(table):
            next_index = same_type.iloc[current_index+2:, 1:].eq(False).idxmax().max()
            if (next_index+2)>=len(table):
                next_index = len(table)
            #Create sub-headers
            sub_headers = table.iloc[current_index,:]
            # Combine main headers with sub headers
            combined_headers = [f"{nh} {sh}" for nh, sh in zip_longest(new_headers, sub_headers, fillvalue="")]
            # Extract the sub-table
            sub_table = table.iloc[current_index+1:next_index]
            # Create the final DataFrame for this section
            ft = pd.DataFrame(sub_table.values, columns=combined_headers)
            # Add to the list of final DataFrames
            final_tables.append(ft)
            # Move to the next section
            current_index = next_index

    return final_tables

final_tables = organize_tables(t2,same_type)

#Convert each table into a list of sentences that relates the treatments to the
#observed responses. Assume that the following sentence structure generally 
#works: "The treatment {header(column 0) } of { cell[next.row,0] } led to a response 
#in { header(next.column) } of {cell[next.row,next.column]}."

def dataframe_to_sentences(table):
    sentences = []
    # Iterate through each row starting from the first row of data
    for row in range(0, table.shape[0]):
        for col in range(1, table.shape[1]):
            treatment = table.columns[0] 
            subject = table.iloc[row, 0]
            response_variable = table.columns[col]
            response_value = table.iloc[row, col]
            if pd.notna(response_value):  # Only create sentences for non-NaN values
                sentence = (f"The treatment type is {treatment}; the treatment ID is {subject}; "
                            f"the response variable is {response_variable}; "
                            f"it produced a response of {response_value}.")
                sentences.append(sentence)
    return sentences

def headers_to_sentences(table):
    sentences = []
    # Iterate through each column
    for col in range(1, table.shape[1]):
        treatment = table.columns[0] 
        response_variable = table.columns[col]
        sentence = (  f"The response variable is {response_variable}.")
        sentences.append(sentence)
    return sentences


#Naive first attempt at this: 
table_sentences = dataframe_to_sentences(final_tables[0])
table_text = " ".join(table_sentences)
doc,entities = extract_entities(table_text)
response_table = create_table(doc, entities, study_id=1)
df = pd.DataFrame(response_table)

#Naive second attempt: 
header_sentence = " ".join(header_sentence)                                                      
doc,entities = extract_entities(header_sentence)

def find_response_cols(table):
    sentences_with_treatment = [] 
    sent_index = []
    treat_name = []
    sent_now = 0
    #Get the response variables from the table by turning them
    #into sentences and passing through the NER
    header_sentence = headers_to_sentences(table)
    header_sentence = " ".join(header_sentence)                                                      
    doc,entities = extract_entities(header_sentence)
    for sent in doc.sents:
        # Check each entity in the doc
        for ent in doc.ents:
            # Check if the entity is within the sentence boundaries and has the label 'TREATMENT'
            if ent.start >= sent.start and ent.end <= sent.end and ent.label_ == "RESPONSE":
                sentences_with_treatment.append(sent.text)
                sent_index.append(sent_now)
                treat_name.append(ent.text)
                break  # Once we find a TREATMENT entity in the sentence, we can move to the next sentence
        sent_now +=1
    return sent_index, sentences_with_treatment, treat_name

#Use the output to grab the correct info from each table and format it and
#convert it to the write format for output (to match the table format from 
#the main text, in extract_responses_txt_v2.py)
study_id = pdf.lstrip('./../papers/').rstrip('.pdf')
column_list = ['STUDY', 'TREATMENT','RESPONSE','CARDINAL','PERCENTAGE','SENTENCE', 'ISTABLE']
final_df = pd.DataFrame(columns = column_list )
row_index = 0
for t1 in final_tables:
    r_index, r_sent, r_name = find_response_cols(t1)
    new_rows = t1.iloc[:, r_index + [(max(r_index)+1)]  ] 
    new_rows.columns.values[0] = 'TREATMENT'
    # Melt the DataFrame so that column names become SENTENCE
    nr_melted = pd.melt(new_rows, id_vars=['TREATMENT'], var_name='SENTENCE', value_name='CARDINAL')
    #Add the standardized (i.e. NER label) name of the response
    # Repeat the labels to match the length of nr_melted
    r_name_long = pd.Series(r_name).repeat(len(nr_melted) // len(pd.Series(r_name)) + 1)[:len(nr_melted)]
    nr_melted['RESPONSE'] = r_name_long.values
    #Add the remaining columns: study id, percentage, istable:
    nr_melted['STUDY'] = study_id
    nr_melted['PERCENTAGE'] =''
    nr_melted['ISTABLE'] = 99 #This obviously came from a table
    final_df=pd.concat([final_df, nr_melted[column_list] ], axis=0 )


#I think a better approach is just going to have the NER look for a response that 
#we want in each table, and then if it exists, figure out which column(s) are relevant 
#to the response and grab those. 
# Iterate through sentences in doc. 
# See whether a certain entity label (i.e. RESPONSE has been found)



#Case 2 and 3: 
# Find the index where the type switch occurs (row with False values)
 # Ignore the first row and first column, then find the first occurrence of False
header_index = same_type.iloc[1:, 1:].eq(False).idxmax().max() 

# Initialize the list to hold the final DataFrames
final_dfs = []

#Case 2: 
# Concatenate rows 0 to header_index to create new headers
header_rows = t2.iloc[:header_index]
new_headers = header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0)

#Case 2: 
# Create the new DataFrame with the remaining rows and the new headers
# data_rows = t2.iloc[header_index + 1:]
# final_df = pd.DataFrame(data_rows.values, columns=new_headers)

#Case 3: 
# Iterate through the table to find additional sub-headers and divide the table into sub-sections
current_index = header_index-1
while current_index < len(t2):
    next_index = same_type.iloc[current_index+2:, 1:].eq(False).idxmax().max()
    # If no more sub-headers are found, process the remaining rows
    if pd.isna(next_index):
        next_index = len(t2)
    
    # Find the current sub-header# 
    #Create sub-headers
    sub_headers = t2.iloc[current_index,:]
    # Combine main headers with sub headers
    combined_headers = [f"{nh} {sh}" for nh, sh in zip_longest(new_headers, sub_headers, fillvalue="")]
    # Extract the sub-table
    sub_table = t2.iloc[current_index+1:next_index]
    # Create the final DataFrame for this section
    final_df = pd.DataFrame(sub_table.values, columns=combined_headers)
    # Add to the list of final DataFrames
    final_dfs.append(final_df)
    # Move to the next section
    current_index = next_index

    
    #sub_headers = sub_header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0)
    




# Clean column names: remove special characters, make lowercase, strip whitespace
t2.columns = [re.sub(r'[\*\?°]', '', col.strip().lower()) for col in t2.columns]

#Identify rows that are potential headers: 



t2 = t2.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)


image_path = "./tables/sample_2.png"
image = cv2.imread(image_path)
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
plt.show()

#analyzer = dd.get_dd_analyzer(config_overwrite=["LANGUAGE='deu'"])


doc=iter(df)
page = next(doc)

type(page)

print(f" height: {page.height} \n width: {page.width} \n file_name: {page.file_name} \n document_id: {page.document_id} \n image_id: {page.image_id}\n")


page.get_attribute_names()

page.document_type, page.language

image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
plt.show()

for layout in page.layouts:
    if layout.category_name=="title":
        print(f"Title: {layout.text}")

page.chunks[0]

table = page.tables[0]
table.get_attribute_names()

print(f" number of rows: {table.number_of_rows} \n number of columns: {table.number_of_columns}" )

table.csv
table.text

cell = table.cells[0]
cell.get_attribute_names()

word = cell.words[0]
word.get_attribute_names()

imported table t1 actually has two header rows as you can see, rows 0 and 1. What would be a way to automatically detect if there are multiple header rows and then combine/concatenate the words into single column headers. So by the end, t1 should end up with two sets of header columns: one with "Plant material_" as a prefix and one with "Total dry weight (g)_", e.g. Plant material_NM, Plant material_AM...etc.?

# Function to automatically detect and combine header rows
def combine_header_rows(table):
    header_rows = []
    combined_headers = []
    for idx, row in table.iterrows():
        header = []
        combined_header = []

        # Iterate over each cell in the row
        for cell in row:
            if isinstance(cell, str) and cell.strip() != '':
                header.append(cell.strip())

        # Combine header elements into a single string
        combined_header = ' '.join(header)

        # Add combined header to list
        combined_headers.append(combined_header)

        # Store original header rows for further processing
        header_rows.append(row)

    # Remove duplicate header rows
    unique_headers = list(set(combined_headers))

    # Initialize dictionary to store combined headers
    combined_dict = {}
    for header in unique_headers:
        combined_dict[header] = []

    # Iterate over the original header rows and populate combined headers
    for row in header_rows:
        for idx, cell in enumerate(row):
            for header in unique_headers:
                if header in cell:
                    combined_dict[header].append(cell)

    # Create DataFrame from combined headers dictionary
    combined_df = pd.DataFrame.from_dict(combined_dict, orient='index').transpose()

    return combined_df





# Iterate through the table to find additional sub-headers and divide the table into sub-sections
current_index = header_index-1
while current_index < len(t2):
    # Find the next sub-header switch
    next_index = same_type.iloc[current_index+1:, 1:].eq(False).idxmax().max()
    # If no more sub-headers are found, process the remaining rows
    if pd.isna(next_index):
        next_index = len(t2)
    
    # Create sub-headers
    sub_header_rows = t2.iloc[next_index,:]
    #sub_headers = sub_header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0)
    # Combine main headers with sub headers
    combined_headers = [f"{nh} {sh}" for nh, sh in zip_longest(new_headers, sub_headers, fillvalue="")]
    # Extract the sub-table
    sub_table = t2.iloc[current_index:next_index]
    # Create the final DataFrame for this section
    final_df = pd.DataFrame(sub_table.values, columns=combined_headers)
    # Add to the list of final DataFrames
    final_dfs.append(final_df)
    # Move to the next section
    current_index = next_index