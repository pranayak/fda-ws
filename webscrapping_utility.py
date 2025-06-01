import requests
import numpy as np
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm

import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def getHTMLContent(link):
    html = urlopen(link)
    soup = BeautifulSoup(html, 'html.parser')
    return soup
    
def extract_informations_from_prim_page (urlinput):
    url_temp = 'https://www.fda.gov'+urlinput
    content_tmp = getHTMLContent(url_temp)
    cc = content_tmp.find('div', {'class':'col-md-8','role':'main'})
    # Extract all <p> tags before the first <h2> - Which is Intro of the study Approved Indication
    current_p_tags = cc.find_all_next('p')
    first_h2_tag = cc.find('h2')
    if cc.div is not None:
        temp_dict = {'Approved Indication': [cc.div.get_text()]}
    elif first_h2_tag is None:
        para1 = [j.text for j in current_p_tags]
        temp_dict = {'Approved Indication': para1}
    else:
        previous_p_tags = first_h2_tag.find_all_previous('p')
        para1 = [j.text for j in current_p_tags if j in  previous_p_tags]

        temp_dict = {'Approved Indication': para1}

        # Extract the content below each <h2> tag
        h2_tags = cc.find_all('h2')
        
        for h2_tag in h2_tags:
            temp_plist = []
            #print(f"Section: {h2_tag.text}")
            next_p_tag = h2_tag.find_next('p')
            while next_p_tag and next_p_tag.find_previous('h2') == h2_tag:
                #print(next_p_tag.text)
                next_p_tag = next_p_tag.find_next('p')
                temp_plist.append(next_p_tag.text)
            temp_dict = {**temp_dict, h2_tag.text:temp_plist}
    return temp_dict
    
def extract_informations_url2 (urlinput):
    url_temp = 'https://www.fda.gov'+urlinput
    #print('URL2 Name:',url_temp)
    content_tmp = getHTMLContent(url_temp)
    cc = content_tmp.find('div', {'class':'col-md-8', 'role':'main'})
    # Extract all <p> tags before the first <h2> - Which is Intro of the study Approved Indication
    current_p_tags = cc.find_all_next('p')
    #print('Len of current page:',len(current_p_tags))
    first_h2_tag = cc.find('h2')
    if first_h2_tag is None:
        para1 = [j.text for j in current_p_tags]
        temp_dict = {'Approved Indication': para1}
    else:
        previous_p_tags = first_h2_tag.find_all_previous('p')
        para1 = [j.text for j in current_p_tags if j in  previous_p_tags]
        temp_dict = {'Approved Indication': para1}
        # Extract the content below each <h2> tag
        h2_tags = cc.find_all('h2')
        for h2_tag in h2_tags:
            temp_plist = []
            #print(f"Section: {h2_tag.text}")
            next_p_tag = h2_tag.find_next('p')
            while next_p_tag and next_p_tag.find_previous('h2') == h2_tag:
                #print(next_p_tag.text)
                next_p_tag = next_p_tag.find_next('p')
                temp_plist.append(next_p_tag.text)
            temp_dict = {**temp_dict, h2_tag.text:temp_plist}
    return temp_dict
  
## Function to Split Brand Name and Active Ingredient
def split_drugname(x, type=1):
    abc = x.split('(')
    if type==1:
        abc1 = abc[0]
        return abc1.strip()
    else:
        abc2 = abc[1]
        abc2 = abc2.replace(')','').strip()
        return abc2

## Function to Split Brand Name and Active Ingredient
def split_drugname_url2(x, type=1):
    abc = x.split('(')
    if type==1:
        abc1 = abc[0]
        return abc1.strip() if len(abc1) > 0 else ''
    else:
        abc2 = abc[1] if len(abc) > 1 else ''
        abc2 = abc2.replace(')','').strip() if len(abc2) > 0 else ''
        return abc2
    
## Function to get Approved Indications    
def get_approved_indication(x):
    if not isinstance(x, list):
        x = [x]
    abc = x[0]
    if len(abc) < 4:
        abc = x[1]
    index_of_the = abc.find("the")
    output = abc[index_of_the:]
    return output

## Function to get Approved Indications    
def get_approved_indication_url2(x):
    output = ''
    for abc in x:
        index_of_the = abc.find("the Food and Drug")
        index_of_the1 = abc.find("he U.S. Food and Drug")
        if (index_of_the > -1):
            output = abc[index_of_the:]
            return output
        if (index_of_the1 > -1):
            output = abc[index_of_the1:]
            return output
    return output

## Function to Extract Company Name
def find_company_name(find_value,x):
    abc = x.lower()
    index_of_the = abc.find(find_value.lower())
    output = x[index_of_the:]
    idx = output.find(' ')
    abc = output[idx:].split(')')[0]
    return abc.strip()
 
    
## Relevant Expedited Programs
def Relevant_Expedited_Programs (a, b, c):
    inputlist = [a, b, c]
    str_op = []
    for a in inputlist:
        for i, j in enumerate(a):
            index = j.lower().find('priority review')
            index1 = j.lower().find('breakthrough therapy')
            index2 = j.lower().find('orphan drug')
            index3 = j.lower().find('accelerated approval')
            index4 = j.lower().find('fast track')
            if index!=-1:
                str_op = str_op+['Priority Review']
            if index1!=-1:
                str_op = str_op+['Breakthrough Therapy Designation']
            if index2!=-1:
                str_op = str_op+['Orphan Drug Designation']
            if index3!=-1:
                str_op = str_op+['Accelerated Approval']
            if index4!=-1:
                str_op = str_op+['Fast Track Desgination']
    str_op = np.unique(str_op).tolist()
    return ';'.join(str_op)   
    
## Function to Extract Company Name
def find_brand_name(x):
    abc2 = x.split(',')
    abc1 = x.split(';')
    abc3 = x.split(' ')
    if len(abc1)>1:
        return abc1[0].strip()
    elif len(abc2)>1:
        return abc2[0].strip()
    elif len(abc3)>1:
        return abc3[0].strip()
    else:
        return x

def get_the_drug_name (a):
    a1 = find_company_name('approval to',a)
    a2 = find_company_name('indication of',a)
    a3 = find_company_name('approved',a)
    a4 = find_company_name('indication for',a)
    a5 = find_company_name('approval of',a)
    if len(a1) > 5:
        return a1.replace('to', '').strip()
    elif len(a2)>5:
        return a2.strip()
    elif len(a3)>5:
        return a3.strip()
    elif len(a4)>5:
        return a4.strip()
    else:
        return ''  
    
def update_bn_ai_cn (config, inputdata):
    dataop = inputdata.copy()
    for key, value in config.items():
        #print(key)
        condition = dataop['Approved_Indication'].str.contains(key, regex=False)
        dataop.loc[condition, ['Brand_Name', 'Active_Ingredient', 'Company_name']] = [value.get('Brand_Name', ''), value.get('Active_Ingredient', ''), value.get('Company_name', '')]
    return dataop

def update_cancertype (config, inputdata):
    dataop = inputdata.copy()
    dataop['Broad_Tumor'] = ''
    for key, value in config.items():
        #print(key)
        for j in value:
            #print(j)
            condition = dataop['Approved_Indication'].str.lower().str.contains(j.lower(), regex=False)
            #print(dataop[condition].shape)
            dataop['Broad_Tumor'] = np.where(condition, key, dataop['Broad_Tumor'])
    return dataop

def update_biomarker (config, x):
    op =  [k for k,v in config.items() if v in x.lower()]
    return ', '.join(list(set(op)))


############# GROK LLMs

def format_text(content):
    """Process text and convert formatting symbols to ReportLab-compatible styles."""
    formatted_lines = []
    
    for section in content.split("\n\n"):  # Ensure paragraph separation
        lines = section.split("\n")  # Split individual lines
        
        for i, line in enumerate(lines):
            line = line.strip()

            # If it's the first line and starts with ###, treat it as a header
            if i == 0 and line.startswith("### "):
                header_text = line[4:].strip()  # Remove "###"
                formatted_lines.append(f'<b><u>{header_text}</u></b>')  # Apply bold + underline formatting
            else:
                # Apply bold formatting only within ** or ***
                line = re.sub(r"\*\*\*(\S.*?)\*\*\*", r'<b>\1</b>', line)  # Handle ***bold***
                line = re.sub(r"\*\*(\S.*?)\*\*", r'<b>\1</b>', line)  # Handle **bold**

                formatted_lines.append(line)  # Keep rest of the text in normal format

        formatted_lines.append("")  # Add spacing after each section

    return formatted_lines

def create_pdf(content, file_name="Formatted_Output.pdf"):
    """Generate a well-formatted PDF with headers, subheaders, bold text, and correct paragraph breaks."""
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    
    normal_style = styles["Normal"]
    header_style = ParagraphStyle(name="Header", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=10)

    elements = []
    formatted_text = format_text(content)

    for line in formatted_text:
        if "<b><u>" in line:  # Subheading
            elements.append(Paragraph(line, header_style))
        else:
            elements.append(Paragraph(line, normal_style))

        elements.append(Spacer(1, 0.1 * inch))  # Adds spacing between paragraphs

    # Build PDF
    doc.build(elements)
    print(f"PDF saved as {file_name}")


# system_prompt =
def create_system_prompt(x):
    std_text = f"Assuming 20% to 30% share of treated patients, what is the Potential Peak Sales for the drug “{x['Active_Ingredient']}” in the in this indication in the US, EU5, China and Japan? And what is the $ value of 1% share of treated patients in these geographies?"
    return std_text

def llm_generator (model_name, header_name, url, system_prompt, user_prompt):
    data = {
    "model": model_name,
    "messages": [{
        "role": "system",
        "content": system_prompt},
        {
        "role": "user",
        "content": user_prompt}
        ]
        }

    response = requests.post(url, headers=header_name, json=data)
    if response.status_code==200:
        return response.json()
    else:
        return ['response error']
    


def llm_generator_v2 (model_name, header_name, url, system_prompt):
    data = {
    "model": model_name,
    "messages": [{
        "role": "system",
        "content": system_prompt}]
        }

    response = requests.post(url, headers=header_name, json=data)
    if response.status_code==200:
        return response.json()
    else:
        return ['response error']