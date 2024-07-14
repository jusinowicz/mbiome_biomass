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

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML
from itertools import zip_longest
import pandas as pd
import numpy as np 
import deepdoctection as dd
import re