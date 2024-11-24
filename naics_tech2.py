#imports
import streamlit as st
from transformers import pipeline
from groq import Groq
from pyppeteer import launch
import subprocess
import sys
import json

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

naics_df = pd.read_csv('NAICS_KEYWORDS.csv')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
naics_embeddings = model.encode(naics_df['2022 NAICS Keywords'].tolist())

def get_relevant_naics(summary, naics_embeddings, naics_df, top_n=5):
    summary_embedding = model.encode([summary])
    similarities = cosine_similarity(summary_embedding, naics_embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    relevant_naics = naics_df.iloc[top_indices]     
    return relevant_naics

#make file name from url
def file_name_from_url(url):
    file_name = url.split("/")[2].replace(".", "").replace('www', '')

    return f"{file_name}_soup.txt"

#list for error urls
errored_urls = []
scrape_list = []

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

                    try:
                        scraped_data = json.loads(result.stdout)
                    except json.JSONDecodeError as e:
                        print("JSONDecodeError:", e)
                        print("Output was not valid JSON. Returning empty list.")
                        scraped_data = []

                #add scraped content to list
                    sources = [data['url'] for data in scraped_data]
                    combined_content = "\n".join([data['content'] for data in scraped_data])

            except Exception as e: 
                # errored_urls.append(url)
                print("\n\nError: could not be scraped because:")
                print(e, "\n\n")

            # Display a loading message
            with st.spinner("NAICS code loading..."):
                api_key = st.secrets["general"]["APIKey"]
                client = Groq(api_key=api_key)

                relevant_naics = get_relevant_naics(combined_content, naics_embeddings, naics_df)

                # Prompt preparation
                website = f"{urls}"
                user_prompt = f"""Given the data scraped from {website}, identify the most probable NAICS code for the company described. 
                The NAICS code must be a valid and existing code from the NAICS hierarchy. 
                Additionally, provide a bulleted list explaining how each part of the NAICS code was derived, including:
                - The main economic sector
                - The subsector
                - The industry group
                - The NAICS industry
                - The national industry
                Provide the explanation using keywords or phrases from the scraped data that contributed to identifying each part of the code. 

                IMPORTANT: 
                - Only predict officially recognized codes from the most recent NAICS classification.
                - Do not include NAICS codes from other countries or regions.
                - Do not fabricate or guess codes or numbers that do not follow the structure.
                - Ensure the code reflects the latest updates to the system.

                Here are some possible NAICS codes with their descriptions:
                {relevant_naics[['2022 NAICS Code', '2022 NAICS Keywords']]}
                
                FORMAT YOUR RESPONSE STRICTLY AS FOLLOWS:

                [Valid NAICS Code]

                Explanation:
                - Main Economic Sector: [Sector] 
                - Subsector: [Subsector]
                - Industry Group: [Industry Group]
                - NAICS Industry: [Industry Name]
                - National Industry: [National Industry]

                No other text or explanation is required.
                """

                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "user", "content": combined_content},
                        {"role": "user", "content": user_prompt}
                        ],
                    temperature=1,
                    max_tokens=1024,
                    stop=None
                )    

                naics_code = ""

                naics_code = completion.choices[0].message.content

                # Display results
                st.success(f"NAICS code determined: {naics_code.strip()}")

        else:
            st.error("Please enter a valid URL starting with http:// or https://")

    #SCRAPING------------------------------------------------------------------------------------------------------


with col2:
    st.header("Instructions", divider="red")
    st.markdown('''### Enter link(s) to website for company you want to classify''')
    st.markdown('''Enter at least one url that links to page(s) with the most information about the company you with to classify.
                Company information is commonly found on "About", "Services", or "Home" pages.
                Sometimes you may need to include more than one url to link to a company's offerings. 
                If including more than one url, separate urls with a comma (example: https://www.naics.com/, https://www.naics.com/search/ ''')
    st.markdown('''### Click the 'Enter' key and wait for a NAICS code to be returned''')
    st.markdown('''This process can take a few minutes. Please be patient and let the app run.''')
    st.markdown('''### NAICS Code is returned!''')
    st.markdown('''You can now close out of any additional pages that opened while the app was running.''')