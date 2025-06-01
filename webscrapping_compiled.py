import requests
import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import subprocess

from webscrapping_utility import getHTMLContent
from webscrapping_utility import extract_informations_from_prim_page
from webscrapping_utility import split_drugname
from webscrapping_utility import get_approved_indication
from webscrapping_utility import find_company_name
from webscrapping_utility import Relevant_Expedited_Programs

# Define the URL of the web page you want to scrape
url = 'https://www.fda.gov/drugs/resources-information-approved-drugs/ongoing-cancer-accelerated-approvals'

# Send an HTTP request to fetch the page content
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all article titles (adjust the CSS selector as needed)
article_titles = soup.select('description')

# Print the titles
# for title in article_titles:
#     print(title.text.strip())

## Load the Content   
content = getHTMLContent(url)
table = content.find('table', {'class': 'table table-striped'})
table.find_all('thead')

## Extract content from URL for each study
master_output = {}
master_output_dummy = {}
tbody = table.find('tbody')
tr_tbody = tbody.find_all_next('tr')
from tqdm import tqdm

for i, study in tqdm(enumerate(tr_tbody), total=len(tr_tbody), desc="Processing URL1"):
    temp_list = []
    ref_title = study.find('a')
    
    if ref_title is None:
        temp_list.extend(['NA'] * 5)
    else:
        hrefs = ref_title.get('href')
        #print('hrefs is : ', hrefs)
        tempdf = extract_informations_from_prim_page(hrefs)
        master_output_dummy[i] = tempdf
        
        temp_list.extend([
            ref_title.get('href'),
            ref_title.get('data-entity-substitution'),
            ref_title.get('data-entity-type'),
            ref_title.get('data-entity-uuid'),
            ref_title.get('title')
        ])

    td_ = study.find_all('td')
    temp_list.extend([j.get_text() for j in td_])
    
    master_output[i] = temp_list

    
### Cleaning the table
col_names = ['href','data_entity_substitution','data_entity_type','data_entity_uuid', 'title', 'Drug_Name', 'AA_Indication', 'AA_Date', 'Original_Projected_Completion', 'AA_Post_Marketing_Requirement1']
tmp_df = pd.DataFrame.from_dict(master_output, orient='index', columns= col_names).reset_index()
tmp_df1 = pd.DataFrame.from_dict(master_output_dummy, orient='index').reset_index()
masterdf = pd.merge(tmp_df, tmp_df1, on='index', how='left')
### Compute Approved Indication
masterdf['Approved Indication'] = np.where(masterdf['Approved Indication'].isna(), masterdf['AA_Indication'], masterdf['Approved Indication'])
print(masterdf.shape)

#masterdf[masterdf['Drug_Name'].str.contains('pralatrexate')]['Approved Indication'].tolist()
tempdf = masterdf.copy()
tempdf['Brand_Name'] = tempdf['Drug_Name'].apply(lambda x: split_drugname(x, type=1))
tempdf['Active_Ingredient'] = tempdf['Drug_Name'].apply(lambda x: split_drugname(x, type=2))
#print(tempdf.head(3))
tempdf['Approved Indication'] = tempdf['Approved Indication'].fillna("NA")

tempdf['Approved_Indication'] = tempdf['Approved Indication'].apply(lambda x: get_approved_indication(x))
tempdf['Company_name'] = tempdf.apply(lambda x: find_company_name(x['Brand_Name'],x['Approved_Indication']), axis=1)

### Create Relevant Expedited Programs
t = masterdf[['index','Approved Indication', 'Efficacy and Safety', 'Expedited Programs']].copy()
t.fillna('NA', inplace=True)
#print(t.shape)
t['Relevant Expedited Programs'] = t.apply(lambda x: Relevant_Expedited_Programs(x['Approved Indication'], x['Efficacy and Safety'], x['Expedited Programs']), axis=1)
tempdf_final = pd.merge(tempdf, t[['Relevant Expedited Programs', 'index']], on='index', how='left')

## Cleaning the Approved Indication column "Removing some special characters
A = '\xa0'
tempdf_final['Approved Indication'] = tempdf_final['Approved Indication'].apply(lambda x: x if isinstance(x, list) else [x])
tempdf_final['Approved Indication'] = tempdf_final['Approved Indication'].apply(lambda x: [j.replace('\xa0', ' ') for j in x if j not in A])

### Adding URL link of the study
tempdf_final['href'] = tempdf_final['href'].apply(lambda x: 'https://www.fda.gov'+x if x != 'NA' else '')

### Final output of the URL 1 
url1_output = tempdf_final[['index', 'AA_Date', 'Drug_Name', 'Brand_Name', 'Active_Ingredient', 'Approved_Indication', 'Company_name','Relevant Expedited Programs', 'Approved Indication', 'Efficacy and Safety', 'Expedited Programs','href']].copy()

url1_output.to_csv("fda_new_approvals_url1_data.csv", index=False)
print("Webscraping Step1 is completed...")
#exec(open("webscrapping_url2.py").read())
#subprocess.run("python webscrapping_url2.py", shell=True, check=True)

## ------------------------------------------------------------------------------------------------------
## Webscrapping Section 2 ------------
##-------------------------------------------------------------------------------------------------------

global latest_approvals

from webscrapping_utility import getHTMLContent
from webscrapping_utility import extract_informations_url2
from webscrapping_utility import split_drugname_url2, get_approved_indication_url2, find_company_name
from webscrapping_utility import find_brand_name, get_the_drug_name, Relevant_Expedited_Programs
from webscrapping_utility import update_bn_ai_cn, update_cancertype, update_biomarker
from webscrapping_utility import create_pdf, format_text, create_system_prompt, llm_generator
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import json
import os
import time

print("Webscraping Step2 is Started...")
## Load Config files
file_path = "config_file.json"
with open(file_path, 'r') as file:
    config_file = json.load(file)

### Load Biomarker and Broad Tumor
file_path = "config_file_cancer_type.json"
with open(file_path, 'r') as file:
    cancer_type = json.load(file)

file_path = "config_file_biomarker_type.json"
with open(file_path, 'r') as file:
    bimarker_type = json.load(file)

## Read - GROK config
## Load Config files
file_path = "config.json"
with open(file_path, 'r') as file:
    config_grok = json.load(file)


# Define the URL of the web page you want to scrape
url = 'https://www.fda.gov/drugs/resources-information-approved-drugs/oncology-cancer-hematologic-malignancies-approval-notifications'

# Send an HTTP request to fetch the page content
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all article titles (adjust the CSS selector as needed)
article_titles = soup.select('description')

#### Testinting Run ----------------------------------------------------
content = getHTMLContent(url)
#table = content.find('table', {'class': 'table table-striped'})
cc = content.find('div', {'class':'col-md-8', 'role':'main'})
#table.find_all('thead')
table = cc.find('table', {'class': 'table table-striped table-border'})
table.find_all('thead')

master_output = {}
master_output_dummy = {}
tbody = table.find('tbody')
tr_tbody = tbody.find_all_next('tr')
from tqdm import tqdm

for i, study in tqdm(enumerate(tr_tbody), total=len(tr_tbody), desc="Processing URL2"):
    temp_list = []
    ref_title = study.find('a')

    if ref_title is None:
        temp_list.extend(['NA'] * 5)  # Add placeholder values
    else:
        try:
            hrefs = ref_title.get('href')
            tempdf = extract_informations_url2(hrefs)  # Attempt extraction
            
            # Update master_output_dummy
            master_output_dummy[i] = tempdf

            temp_list.extend([
                hrefs,
                ref_title.get('data-entity-substitution'),
                ref_title.get('data-entity-type'),
                ref_title.get('data-entity-uuid'),
                ref_title.get('title')
            ])

        except Exception as e:
            print(f"Error processing href: {hrefs} - {e}")
            continue  # Skip iteration if error occurs

    # Extract table data
    td_ = study.find_all('td')
    temp_list.extend([j.get_text() for j in td_])

    # Update master_output dictionary
    master_output[i] = temp_list


## Convert dictionary into Pandas Dataframe for all extracted fields
col_names = ['href','data_entity_substitution','data_entity_type','data_entity_uuid', 'title', 'Webpage', 'Description', 'Date']
tmp_df = pd.DataFrame.from_dict(master_output, orient='index', columns= col_names).reset_index()
tmp_df1 = pd.DataFrame.from_dict(master_output_dummy, orient='index').reset_index()
masterdf = pd.merge(tmp_df, tmp_df1, on='index', how='left')
print(masterdf.shape)
masterdf.to_csv('webscrape_step2_raw_data_df.csv', index=False)


#### Testinting Run ---------------------------------------------------- end 
#masterdf = pd.read_csv('webscrape_step2_raw_data_df.csv')

### Extract Approved Indication 
tempdf = masterdf.copy()
A = '\xa0'
tempdf['Description'].fillna("NA", inplace=True)
tempdf['Approved Indication'].fillna("NA", inplace=True)
tempdf['Approved Indication'] = tempdf['Approved Indication'].apply(lambda x: x if isinstance(x, list) else [x])
tempdf['Approved Indication'] = tempdf['Approved Indication'].apply(lambda x: [j.replace('\xa0', ' ') for j in x if j not in A])
tempdf['Approved_Indication'] = tempdf['Approved Indication'].apply(lambda x: get_approved_indication_url2(x))

### Extract Drug Name, Company Name and Brand Name
tempdf['Drug_Name'] = tempdf['Approved_Indication'].apply(lambda x: get_the_drug_name(x))
tempdf['Active_Ingredient'] = tempdf['Drug_Name'].apply(lambda x: split_drugname_url2(x, type=1))
tempdf['Brand_Name'] = tempdf['Drug_Name'].apply(lambda x: split_drugname_url2(x, type=2))
tempdf['Brand_Name'] = tempdf['Brand_Name'].apply(lambda x: find_brand_name(x))
tempdf['Company_name'] = tempdf.apply(lambda x: find_company_name(x['Brand_Name'],x['Drug_Name']), axis=1)
tempdf['Company_name'] = tempdf.apply(lambda x: find_company_name(x['Brand_Name'],x['Drug_Name']), axis=1)

### Extract Relevant Expedited Programs
t = tempdf[['index','Approved Indication', 'Efficacy and Safety', 'Expedited Programs']].copy()
t.fillna('NA', inplace=True)
t.shape
t['Relevant Expedited Programs'] = t.apply(lambda x: Relevant_Expedited_Programs(x['Approved Indication'], x['Efficacy and Safety'], x['Expedited Programs']), axis=1)

tempdf_final = pd.merge(tempdf, t[['Relevant Expedited Programs', 'index']], on='index', how='left')
tempdf_final['url'] = url


## Update Company Name, Drug Name and Brand Name
tempabc = update_bn_ai_cn (config_file, tempdf_final)

## Update Drug Name
tempabc['Drug_Name'] = tempabc[['Brand_Name','Active_Ingredient']].apply(lambda x: ''+str(x['Brand_Name'])+' ('+str(x['Active_Ingredient'])+')', axis=1)
condition = tempabc['Brand_Name']==''
tempabc.loc[condition, ['Drug_Name', 'Brand_Name', 'Active_Ingredient']] = ['','','']

print('Webscrapping for URL2 is successful... ')

### combine URL 1 and URL2 Data ------------------------------------------------------------------------------
#df_final1 = pd.read_csv('fda_new_approvals_url1_data.csv')
df_final1 = url1_output.copy()
url1 = 'https://www.fda.gov/drugs/resources-information-approved-drugs/ongoing-cancer-accelerated-approvals'
df_final1['url'] = url1
df_final1.columns

df_final11 = update_bn_ai_cn (config_file, df_final1)
tempabc.rename(columns={'Date':'AA_Date'}, inplace=True)

### Append Data 1 and Data2 
master_df = pd.concat([df_final11, tempabc[df_final1.columns]], axis=0)
master_df['url'].value_counts()
## Extract Cancer Type
master_df1 = update_cancertype (cancer_type, master_df)
## Extract Biomarker Name
master_df1['Biomarker'] = master_df1['Approved_Indication'].apply(lambda x: update_biomarker (bimarker_type, x))
## Rename columns
master_df1.rename(columns = {'Approved Indication' : 'Approval_Details', 'href':'url_name'}, inplace=True)
## Adjust Company name where it not available
master_df1['Company_name'] = np.where(len(master_df1['Company_name'])<3,'', master_df1['Company_name'])
## Remove special character only value from Approved Indication
master_df1['Approved_Indication'] = np.where(len(master_df1['Approved_Indication'])==')','', master_df1['Approved_Indication'])
master_df1['Company_name'].fillna('', inplace=True)
master_df1['Approved_Indication'].fillna('', inplace=True)
master_df1['Company_name'] = master_df1['Company_name'].apply(lambda x : x if len(x)>3 else '')
master_df1['Approved_Indication'] = master_df1['Approved_Indication'].apply(lambda x : x if x !=')' else '')


## Format Date column
master_df1['format_date'] = master_df1['AA_Date'].str.replace('\u00a0', '').str.strip()
#master_df1['AA_Date'].apply(lambda x: [j.replace('\u00a', ' ') for j in x])
#master_df1['format_date1'] = master_df1['AA_Date'].apply(lambda x: [j.replace('\u00a', ' ') for j in x])
master_df1['AA_Date'] = pd.to_datetime(master_df1['format_date'], format='%m/%d/%Y')
final_col = ['index','AA_Date','Drug_Name','Brand_Name','Active_Ingredient','Approved_Indication','Company_name','Relevant Expedited Programs','Approval_Details','Efficacy and Safety','Broad_Tumor','Biomarker','url_name']

final_table = master_df1[~master_df1['url_name'].isna()].drop_duplicates(subset=['Active_Ingredient','Approved_Indication'], keep='first')
final_table = final_table[final_col].copy()
final_table = final_table.sort_values("AA_Date", ascending=False)
final_table = final_table.reset_index(drop=True)


final_table.to_csv('fda_new_approvals_combined_data_master.csv', index=False)
#master_df1[final_col].to_json('fda_new_approvals_combined_data_master.json', index=False)
final_table.to_json('final_output_json_master.json', orient = "records")

## Update Study ID ---------------------------
import pickle
with open('study_id_referece.pkl', 'rb') as f:
    data_study_id = pickle.load(f)

uid_list = list(set(zip(final_table['AA_Date'],final_table['url_name'])))
print(len(uid_list))

final_table['test_date'] = final_table['AA_Date'].dt.strftime("%Y-%m-%d")
#final_table['test_date'] = final_table['AA_Date']
final_table['keys_tuple'] = final_table.apply(lambda x: (x['test_date'], x['url_name']), axis=1)
## Add study ID
print(final_table.shape)
final_table['Study_id'] = final_table.apply(lambda x: data_study_id.get(x['keys_tuple'], 'Study_99999'), axis=1)
final_table['Study_id_num'] = final_table.apply(lambda x: int(x['Study_id'].split('_')[1]), axis=1)
max_cnt = final_table[final_table['Study_id_num']<99999]['Study_id_num'].max()
llm_input_df1 = final_table[final_table['Study_id_num']==99999].reset_index(drop=True)
llm_input_df2 = final_table[final_table['Study_id_num']<99999].reset_index(drop=True)
print('llm_input_df1: ',llm_input_df1.shape)
print('llm_input_df2: ', llm_input_df2.shape)

## Generate LLMs for only new Studies
if llm_input_df1.shape[0]>0:
    print('New study exists :', llm_input_df1.shape[0])
    ### LLM setup
    api_key = config_grok['API_Key']
    url_grok = config_grok['URL']
    header_grok = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    grok_model = config_grok['Model_Name']

    ## Define new default Study name
    llm_input_df1['Study_id']='Study_99999'
    llm_input_df1['Study_id_num']=99999
    ## Define Prompt
    llm_input_df1['system_prompt'] = llm_input_df1.apply(lambda x: create_system_prompt(x), axis=1)

    size_n = llm_input_df1.shape[0]
    index_list = llm_input_df1.index
    for j in range(size_n):
        print(j)
        ii = j+1
        study_id = 'Study_'+str(max_cnt+ii)
        print('study_id is :', study_id)
        llm_input_df1.at[index_list[j], 'Study_id'] = study_id
        llm_input_df1.at[index_list[j], 'Study_id_num'] = max_cnt+ii
        update_dict_key = llm_input_df1.at[index_list[j], 'keys_tuple']
        data_study_id[update_dict_key] = study_id
        ## Generate LLM output
        system_prompt = llm_input_df1.loc[index_list[j]]['system_prompt']
        user_prompt = llm_input_df1.loc[index_list[j]]['Approved_Indication']
        llmoutput = llm_generator(grok_model, header_grok, url_grok, system_prompt, user_prompt)
        
        ## Save ouptut in PDF
        curr_path = os.getcwd()
        #filename = os.path.join(curr_path,f"/static/pdfs/{study_id}.pdf")
        filename = os.path.join(f"{curr_path}\\static\\pdfs\\{study_id}.pdf")
        news_summary1 = llmoutput.get('choices')[0]
        if isinstance(news_summary1, dict):
            final_news = news_summary1.get('message').get('content')
            create_pdf(final_news, filename)
            time.sleep(5)
        
    llm_input_df = pd.concat([llm_input_df2,llm_input_df1], axis=0)

    ### Update the study ID
    with open('study_id_referece.pkl', 'wb') as f:
        pickle.dump(data_study_id, f)
else:
    llm_input_df = llm_input_df2.copy()


# Convert to formatted string before passing to HTML -------------------------
# Calculate the date 12 months ago
from datetime import datetime, timedelta
twelve_months_ago = datetime.today() - timedelta(days=365)
# Filter the DataFrame
llm_input_df['pdf_link'] = llm_input_df.apply(lambda x: x['Study_id']+'.pdf', axis=1)
final_col_v1 = ['pdf_link','index','AA_Date','Drug_Name','Brand_Name','Active_Ingredient','Approved_Indication','Company_name','Relevant Expedited Programs','Approval_Details','Efficacy and Safety','Broad_Tumor','Biomarker','url_name']
llm_input_df['AA_Date'] = pd.to_datetime(llm_input_df['AA_Date'])
filtered_df = llm_input_df[llm_input_df['AA_Date'] >= twelve_months_ago][final_col_v1]
filtered_df = filtered_df.sort_values("AA_Date", ascending=False)
filtered_df["AA_Date"] = filtered_df["AA_Date"].dt.strftime("%Y-%m-%d")
filtered_df.to_json('final_output_json_master_sample.json', orient = "records")
print('Fianl Samples size', filtered_df.shape)

print('Webscrapping is successful... ')