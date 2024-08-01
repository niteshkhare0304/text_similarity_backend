import streamlit as st
import openai
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib3
import requests

st.title("Text Similarity Checker")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Monkey-patch requests to disable SSL verification for OpenAI
old_request = requests.Session.request

def new_request(*args, **kwargs):
    kwargs['verify'] = False  # Disable SSL verification
    return old_request(*args, **kwargs)

requests.Session.request = new_request

# Upload the store and input Excel files
store_file = st.file_uploader("Upload the store Excel file", type=["xlsx"], key="store_file")
input_file = st.file_uploader("Upload the input Excel file", type=["xlsx"], key="input_file")

# Add a slider for similarity threshold
SIMILARITY_THRESHOLD = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)

if store_file and input_file:
    try:
        # Read the Excel files
        store_df = pd.read_excel(store_file)
        input_df = pd.read_excel(input_file)

        # Function to get embeddings from OpenAI
        def get_embedding(text):
            response = openai.Embedding.create(
                input=[text],  # The input should be a list of strings
                model="text-embedding-ada-002"
            )
            return np.array(response['data'][0]['embedding'])

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1)
            vec2 = torch.tensor(vec2)
            dot_product = torch.dot(vec1, vec2).item()
            magnitude = torch.norm(vec1).item() * torch.norm(vec2).item()
            similarity = dot_product / (magnitude + 1e-8)  # Add a small epsilon to avoid division by zero
            return similarity

        # Process each row in the input DataFrame
        most_similar_texts = []
        similarity_scores = []
        for input_index, input_row in input_df.iterrows():
            input_text = input_row['text']

            # Get the embedding for the input text
            input_embedding = get_embedding(input_text)

            # Compare with each text in the store DataFrame
            highest_similarity = 0
            most_similar_text = ""
            for store_index, store_row in store_df.iterrows():
                store_text = store_row['text']
                store_embedding = get_embedding(store_text)

                similarity = cosine_similarity(input_embedding, store_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_text = store_text

            most_similar_texts.append(most_similar_text)
            similarity_scores.append(highest_similarity)

        # Add the results to the input DataFrame
        input_df['most_similar_text'] = most_similar_texts
        input_df['similarity_score'] = similarity_scores

        # Save the result to a new Excel file
        output_file = "similarity_results.xlsx"
        input_df.to_excel(output_file, index=False)

        # Provide a download link for the output file
        with open(output_file, "rb") as file:
            st.download_button(label="Download Results", data=file, file_name=output_file)

        # Visualize the distribution of similarity scores
        st.subheader("Similarity Score Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(similarity_scores, bins=20, kde=True)
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Similarity Scores")
        st.pyplot(plt.gcf())

        # (Optional) Heatmap of Similarity Scores (if practical for your dataset size)
        # You can implement this part if you want to show the similarity matrix
        # across all inputs and store texts. It's generally useful for smaller datasets
        # to avoid overwhelming the visualization.

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")

    except ValueError as e:
        st.error(f"Error processing file: {e}")
