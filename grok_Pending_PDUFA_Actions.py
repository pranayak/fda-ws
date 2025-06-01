import requests
import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import os
import time

global latest_approvals

from webscrapping_utility import getHTMLContent
from webscrapping_utility import extract_informations_url2
from webscrapping_utility import split_drugname_url2, get_approved_indication_url2, find_company_name
from webscrapping_utility import find_brand_name, get_the_drug_name, Relevant_Expedited_Programs
from webscrapping_utility import update_bn_ai_cn, update_cancertype, update_biomarker

from webscrapping_utility import create_pdf, format_text, create_system_prompt, llm_generator_v2


import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

## Read - GROK config
## Load Config files
file_path = "config.json"
with open(file_path, 'r') as file:
    config_grok = json.load(file)

api_key = config_grok['API_Key']
url_grok = config_grok['URL']
header_grok = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
grok_model = config_grok['Model_Name']

system_prompt = 'Provide a list of drugs with pending PDUFA action dates from the U.S. FDA'
llmoutput_v2 = llm_generator_v2 (grok_model, header_grok, url_grok, system_prompt)

## Save ouptut in PDF
curr_path = os.getcwd()
filename = os.path.join(f"{curr_path}\\static\\pdfs\\Pending_PDUFA_Actions.pdf")
news_summary2 = llmoutput_v2.get('choices')[0]
final_news2 = news_summary2.get('message').get('content')
create_pdf(final_news2, filename)
time.sleep(5)