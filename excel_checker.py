import streamlit as st
import openai
import numpy as np
import torch
import requests
import urllib3
import pandas as pd

st.title("Text Similarity Checker")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set your OpenAI API key
#openai.api_key = 

# Monkey-patch requests to disable SSL verification for OpenAI
old_request = requests.Session.request

def new_request(*args, **kwargs):
    kwargs['verify'] = False  # Disable SSL verification
    return old_request(*args, **kwargs)

requests.Session.request = new_request

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Add a slider for similarity threshold
SIMILARITY_THRESHOLD = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)

if uploaded_file:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Function to get embeddings from OpenAI
        def get_embedding(text):
            response = openai.Embedding.create(
                input=[text],  # The input should be a list of strings
                model="text-embedding-ada-002"  # Change to the desired model
            )
            return np.array(response['data'][0]['embedding'])

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1)
            vec2 = torch.tensor(vec2)
            dot_product = torch.dot(vec1, vec2).item()
            magnitude = torch.norm(vec1).item() * torch.norm(vec2).item()
            similarity = dot_product / (magnitude + .2)  # Add a small epsilon to avoid division by zero
            return similarity

        # Process each row
        similarity_scores = []
        results = []
        for index, row in df.iterrows():
            text1 = row['text1']
            text2 = row['text2']
            embedding1 = get_embedding(text1)
            #print (embedding1)
            #print (embedding2)
            embedding2 = get_embedding(text2)
            similarity = cosine_similarity(embedding1, embedding2)
            similarity_scores.append(similarity)
            results.append(similarity > SIMILARITY_THRESHOLD)

        # Add the results to the DataFrame
        df['similarity_score'] = similarity_scores
        df['text_result'] = results

        # Save the result to a new Excel file
        output_file = "similarity_results.xlsx"
        df.to_excel(output_file, index=False)

        # Provide a download link for the output file
        with open(output_file, "rb") as file:
            st.download_button(label="Download Results", data=file, file_name=output_file)

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")

    except ValueError as e:
        st.error(f"Error processing file: {e}")
