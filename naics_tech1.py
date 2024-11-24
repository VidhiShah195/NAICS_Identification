#imports
import streamlit as st
import re
import time
from bs4 import BeautifulSoup
# from selenium import webdriver
import re, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import os
# from dotenv import load_dotenv
from groq import Groq
from pyppeteer import launch
import asyncio
import subprocess
import sys
import json
# import nest_asyncio
# nest_asyncio.apply()

#progress bar:
# def progress_bar(progress):

#     progress_bar_text = progress_text
#     my_bar=st.progress(0,text=progress_bar)
#     success_bar_text = success_text

#     for percent_complete in range(100):
#         time.sleep(0.01)
#         my_bar.progress(percent_complete+1, text=progress_bar_text)
#     time.sleep(1)
#     print(success_bar_text)


#SCRAPER FUNCTIONS/IMPORTS/COMPONENTS------------------------------------------------------------------------------

#define driver
# driver = webdriver.Firefox()

#get soup for page
# def soupify_url(url, driver=driver):
#     driver.get(url)
#     return BeautifulSoup(driver.page_source, 'html.parser')

#scraper
# def scraper(link, driver=driver):
#     print(f"Scraping {link}")

#     driver.get(link)
#     time.sleep(2)

#     html = driver.page_source
#     raw_soup = BeautifulSoup(html, features="html.parser")
    
#     paragraphs = raw_soup.find_all('p')

#     return " ".join([p.get_text(" ", strip=True) for p in paragraphs])


#make file name from url
def file_name_from_url(url):
    file_name = url.split("/")[2].replace(".", "").replace('www', '')

    return f"{file_name}_soup.txt"

#list for error urls
errored_urls = []
scrape_list = []
#-----------------------------------------------------------------------------------------------------------------
#SUMMARIZER FUNCTIONS/COMPONENTS----------------------------------------------------------------------------------
# def summarize(text):
#     model_name = "allenai/led-large-16384-arxiv"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

#     result = pipe(
#         text,
#         truncation=True, 
#         max_length=750, 
#         no_repeat_ngram_size=5, 
#         num_beams=3, 
#         early_stopping=True
#         )
    
#     summary = result[0]['generated_text']
#     return summary

#STREAMLIT--------------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Determine a company's NAICS code",)
col1, col2 = st.columns(2)

with st.sidebar:
    st.markdown('''# North American Insurance Classification System (NAICS)''')
    st.markdown('''information about naics codes, what they are, what they do, etc''')

urls = []

with col1:
    st.header("Enter Company Information", divider="red")
    company_url = st.text_input("Enter link(s) with company information here:")
    urls.append(company_url)

    # Validate the URL
    if company_url:
        if company_url.startswith("http://") or company_url.startswith("https://"):
            st.success(f"Valid URL: {company_url}")

            # for url in urls:
            try:
                with st.spinner("Scraping Page(s)"):
                    # st.write('starting')
                    python_executable = sys.executable
                    result = subprocess.run([python_executable, 'scraper.py', json.dumps(urls)], capture_output=True, text=True)
                    # processed_soup = asyncio.run(scraper(url))
                    # st.write('finished scraping')

                    try:
                        scraped_data = json.loads(result.stdout)
                    except json.JSONDecodeError as e:
                        print("JSONDecodeError:", e)
                        print("Output was not valid JSON. Returning empty list.")
                        scraped_data = []

                #add scraped content to list
                # scrape_list.append(processed_soup)
                    sources = [data['url'] for data in scraped_data]
                    combined_content = "\n".join([data['content'] for data in scraped_data])

            except Exception as e: 
                # errored_urls.append(url)
                print("\n\nError: could not be scraped because:")
                print(e, "\n\n")
            #show all error urls
            # print("Errored urls:")
            # for url in errored_urls:
            #     print(url)
            # driver.close()

            # st.write('starting output')

            # with st.spinner("Summarizing Information"):
            #     # summarized = summarize(text=scrape_list)
            #     # summarized = summarize(text=combined_content)
            # Display a loading message
            with st.spinner("NAICS code loading..."):
                # load_dotenv()
                # api_key = os.getenv("APIkey")
                api_key = st.secrets["general"]["APIkey"]
                client = Groq(api_key=api_key)

                # Prompt preparation
                website = f"{urls}"
                user_prompt = f"""
                    Given the data that was scraped from {website}, identify the most probable NAICS code and NAICS title. ONLY PROVIDE THE NAICS CODE AND A NAICS CODE TITLE, NO OTHER TEXT. 
                """

                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        # {"role": "user", "content": summarized},
                        {"role": "user", "content": combined_content},
                        {"role": "user", "content": user_prompt}
                        ],
                    temperature=1,
                    max_tokens=24,
                    # top_p=1,
                    # stream=True,
                    stop=None
                )    

                naics_code = ""

                naics_code = completion.choices[0].message.content

                # for chunk in completion:
                #     naics_code += chunk.choices[0].delta.content or ""
                
                # Display results
                st.success(f"NAICS code determined: {naics_code.strip()}")

                # time.sleep(4)  # Simulate a loading process (e.g., fetching NAICS code)
            
            # st.success("NAICS code determined!")

        else:
            st.error("Please enter a valid URL starting with http:// or https://")

    #SCRAPING------------------------------------------------------------------------------------------------------


with col2:
    st.header("Instructions", divider="red")
    st.markdown('''
                ***Please do not close the Firefox window that opens with this streamlit app until NAICS code is returned***''')
    st.markdown('''### Enter link(s) to website for company you want to classify''')
    st.markdown('''Enter at least one url that links to page(s) with the most information about the company you with to classify.
                Company information is commonly found on "About", "Services", or "Home" pages.
                Sometimes you may need to include more than one url to link to a company's offerings. 
                If including more than one url, separate urls with a comma (example: https://www.naics.com/, https://www.naics.com/search/ ''')
    st.markdown('''### Click the 'Enter' key and wait for a NAICS code to be returned''')
    st.markdown('''This process can take a few minutes. Please be patient and let the app run.''')
    st.markdown('''### NAICS Code is returned!''')
    st.markdown('''You can now close out of any additional pages that opened while the app was running.''')