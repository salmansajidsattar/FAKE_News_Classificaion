import streamlit as st
import sqlite3
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import random
import time

load_dotenv()
## Configure Genai Key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
source_names=["al_jazeera,Fox_News,RT_News,APPS_News"]
## Define Your Prompt
prompt=[
    """
    You are an expert in generating the fake news from the given 
    news text along with likes,comments,shares,source . An english text will be passed and you will convert this 
    into a fake news. The generated fake news should be in English language.The generated fake news
    should have the same characters length as orginal news contain.only return the fake news responce.
    
    For example,
    Example 1 - There is increase in the petroleum prices?, 
    output- the fake news will be something like this there is no increase in the petroleum prices,
    """
]
## Function To Load Google Gemini Model and provide queries as response
def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

def Main(df):
    count=0
    data=[]
    for index, row in df.iterrows():
        question=row['text']
        if count==2000:
            break
        count+=1
        try:
            response=get_gemini_response(question, prompt)
            time.sleep(2)
            data.append([str(response),random.randrange(5, 10, 2), random.randrange(5,25,3), random.randrange(1,15,1),random.choice(source_names),0])
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
        print(data)
    return pd.DataFrame(data, columns=['text', 'likes', 'comments', 'shares', 'source', 'label'])

# question="This exclusive footage obtained by Al Jazeera shows the extent of the destruction of a residential area in the Sheikh Radwan neighbourhood, northwest of Gaza City."
# response=get_gemini_response(question,prompt)

df = pd.read_csv("combined_News_output.csv")
out=Main(df)
pd.concat([df, out], ignore_index=True).to_csv("combined_News_output.csv", index=False)
# print(type(response))
# print(response)