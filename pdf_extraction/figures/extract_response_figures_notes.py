#==============================================================================
# For the meta analysis and database: 
# This is STEP 5 in the pipeline:
# 
# Use the library DeepDoctection to extract tables from a scentific paper and 
# convert them into Data Frames (pandas). 
#
# This is meant to be the 2nd step in paper parsing, which goes to the tables to 
# to look for response variables and extract the relevant information. This is the 
# last step before trying more complex figure-extraction methods. 
# 
# Deepdoctection is its own pipeline that is highly customizable. Please check 
# out the useful project notebooks for tutorials: 
# https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb
# 
# Getting Deepdoctection requires some additional installations that may be 
# OS/platform-specific. 
# Needs: torchvision timm
# 
# Once a table is uploaded, the same custom NER built in Spacy is used to look 
# for the response variables of interest. 
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
# See extrat_abstract_dat for further dependency installation notes.  
#
#==============================================================================
py -3.8

#General usage, visulatization
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML
from itertools import zip_longest
import pandas as pd
import numpy as np 
import os

#Deepdoctection 
import deepdoctection as dd
import re

#NER and NLP
import spacy

#==============================================================================
#Set paths: 
#==============================================================================

#Path to PDFs 
pdf_dir = "./../papers/"
#pdf_path = "./../papers/21222096.pdf"
#pdf_path = "./../papers/38948452.pdf"
#pdf_path = "./../papers/30285730.pdf"
#pdf_path = "./../papers/21624119.pdf"

#Current NER to use: 
output_dir = "./../models/custom_web_ner_abs_v382"

# Set the option to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#==============================================================================
# Custom functions: 
#==============================================================================

#==============================================================================
# 1. Start the DD analyzer
# For each PDF: 
# 2. Scan all of the pages,
# 3. Extract Figures
# 4. 
#==============================================================================

# 1. DD analyzer #default THRESHOLD_ROWS: 0.4
#analyzer = dd.get_dd_analyzer(config_overwrite = ["SEGMENTATION.THRESHOLD_ROWS=0.01"])
analyzer = dd.get_dd_analyzer()

# Get the list of current PDFs in the directory
new_pdfs = {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

study_id = pdf_path.lstrip('./../papers/').rstrip('.pdf')
#Initialize the dd analyzer
df = analyzer.analyze(path=pdf_path)
# This method must be called just before starting the iteration. It is part of the API.    
df.reset_state()  
pages =[] #Get the pages in the PDF
for doc in df: 
    pages.append(doc)

figures = pages[].get_annotation(category_names=dd.LayoutType.figure)
for fig in figures:
    fig.viz(interactive=True)  # vizualize the figure with an interactive window



#From https://github.com/deepdoctection/deepdoctection/discussions/280
#Running it this way grabs figures as the text is parsed without storing
#full pages. 

for dp in df:
    figures = dp.get_annotation(category_names=dd.LayoutType.figure)
    for fig in figures:
        fig.viz(interactive=True)  # vizualize the figure with an interactive window
        np_array = fig.viz() # get the numpy array of the figure region
        dd.viz_handler.write_image(f"/path/to/dir/{fig.annotation_id}.png", np_array) # save the numpy array as .png


# Process the PDFs
#Structure of final tables
column_list = ['STUDY', 'TREATMENT','RESPONSE','CARDINAL','PERCENTAGE','SENTENCE', 'ISTABLE']
data = pd.DataFrame(columns = column_list ) #Initialize final table
for pdf in new_pdfs:
    #1.Get PDFs and run through dd
    pdf_path = pdf_dir + pdf
    print(f"Current PDF: {pdf_path}")
    study_id = pdf_path.lstrip('./../papers/').rstrip('.pdf')
    #Initialize the dd analyzer
    df = analyzer.analyze(path=pdf_path)
    # This method must be called just before starting the iteration. It is part of the API.    
    df.reset_state()  
    pages =[] #Get the pages in the PDF
    for doc in df: 
        pages.append(doc)
	#2. Cycle through tables in the pages, look for responses, extract them if they are 
	#there:
    table_num = 1
    for pg in pages:
        for tbls in pg.tables:
            print(f"Table: {table_num}")
            t2 = pd.DataFrame(tbls.csv)
            #Replace blank space with NaN
            t2.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            #Drop columns and rows of NaN created by spaces
            t2.dropna(axis=1, how='all', inplace=True)
            t2.dropna(axis=0, how='all', inplace=True)
            t2 = t2.reset_index(drop=True)  
            # Remove letters and special symbols from numbers
            # Remove leading/trailing whitespace from all cells and make all 
            # lowercase
            t2= t2.applymap(clean_numbers)
            # Check the case where there are just a few straggler entries  in 
            # the middle or at the end of a row. Usually means a word connects to 
            # the cell above it.
            t2 = merge_danglers(t2)
            #Check rows with NaN and handle accordingly. This step is designed
            #to help identify headers/subheaders which might only fill a single cell
            for index, row in t2.iterrows():
                if (row.isna().sum() ) >= len(t2.columns)/2+1:
                    #Fill NaN with previous cell in row
                    row = row.fillna(method='ffill')
                    t2.iloc[index,:] = row
            # Apply the classification function to each cell in the DataFrame
            classified_t2 =classify_cells(t2)  
            #Check whether a row is the same type as previous row
            same_type = is_same_type(classified_t2)
            #Use this information to find column headers and parse the table
            #Try to infer which rows are likely to contain headers based on 
            #where the data type across a row changes. If there seem to be 
            #multiple header rows then divide the table into multiple tables. 
            final_tables = organize_tables(t2,same_type)
            #Use the output to grab the correct info from each table and format it and
            #convert it to the write format for output (to match the table format from 
            #the main text, in extract_responses_txt_v2.py)
            final_df = make_final_table(final_tables, study_id)
            final_df = final_df.reset_index(drop=True)
            print(f"final_df{final_df}")
            data = pd.concat([data, final_df[column_list] ], axis=0 )
            table_num +=1
 
# flattened_data = [item for sublist in data for item in sublist]
# all_tables = pd.DataFrame(flattened_data)

# Export DataFrame to a CSV file
data.to_csv('./../output/extract_from_tables1.csv', index=False)


#What if the data frame is transposed so that the response is along 
#the row headings? 
t2t = t2.transpose()
# Reset the index and set the first row as new column headers
#t2t.columns = t2t.iloc[0]
#t2t = t2t[1:]
t2t.reset_index(drop=True)
# Apply the classification function to each cell in the DataFrame
classified_t2t = classify_cells(t2t)  
#Check whether a row is the same type as previous row
same_typet = is_same_type(classified_t2t)
#Use this information to find column headers and parse the table
#Try to infer which rows are likely to contain headers based on 
#where the data type across a row changes. If there seem to be 
#multiple header rows then divide the table into multiple tables. 
final_tablest = organize_tables(t2t,same_typet)
final_dft = make_final_table(final_tablest, study_id)
#Add this to the growing list of final tables

data = pd.concat([data, final_df[column_list] ], axis=0 )
