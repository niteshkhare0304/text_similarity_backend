import streamlit as st
import openai
import numpy as np
import torch
import requests
import urllib3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm

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
store_file = st.file_uploader("Upload the store Excel file (should contain a 'text' column)", type=["xlsx"], key="store_file")
input_file = st.file_uploader("Upload the input Excel file (should contain a 'text' column)", type=["xlsx"], key="input_file")

# Add a slider for similarity threshold
SIMILARITY_THRESHOLD = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)

def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

def cosine_similarity(vec1, vec2):
    vec1 = torch.tensor(vec1)
    vec2 = torch.tensor(vec2)
    dot_product = torch.dot(vec1, vec2).item()
    magnitude = torch.norm(vec1).item() * torch.norm(vec2).item()
    similarity = dot_product / (magnitude + 1e-8)  # Add a small epsilon to avoid division by zero
    return similarity

if store_file and input_file:
    try:
        with st.spinner("Processing files..."):
            # Read the Excel files
            store_df = pd.read_excel(store_file)
            input_df = pd.read_excel(input_file)

            if store_df.empty or input_df.empty:
                st.error("One of the uploaded files is empty. Please upload valid files.")
            elif 'text' not in store_df.columns or 'text' not in input_df.columns:
                st.error("Both files must contain a 'text' column.")
            else:
                # Calculate embeddings for store texts
                store_embeddings = {}
                with st.spinner("Calculating embeddings for store texts..."):
                    for _, row in tqdm(store_df.iterrows(), total=store_df.shape[0], desc="Processing store texts"):
                        store_text = row['text']
                        store_embeddings[store_text] = get_embedding(store_text)

                # Process each row in the input DataFrame
                most_similar_texts = []
                similarity_scores = []
                with st.spinner("Calculating similarities..."):
                    for _, input_row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Processing input texts"):
                        input_text = input_row['text']
                        input_embedding = get_embedding(input_text)

                        highest_similarity = 0
                        most_similar_text = ""
                        for store_text, store_embedding in store_embeddings.items():
                            similarity = cosine_similarity(input_embedding, store_embedding)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                most_similar_text = store_text

                        most_similar_texts.append(most_similar_text)
                        similarity_scores.append(highest_similarity)

                # Add the results to the input DataFrame
                input_df['most_similar_text'] = most_similar_texts
                input_df['similarity_score'] = similarity_scores

                # Filter results by similarity threshold
                filtered_df = input_df[input_df['similarity_score'] >= SIMILARITY_THRESHOLD]

                # Save the result to a new Excel file
                output_file = "similarity_results.xlsx"
                filtered_df.to_excel(output_file, index=False)

                # Provide a download link for the output file
                with open(output_file, "rb") as file:
                    st.download_button(label="Download Results", data=file, file_name=output_file)

                # Data visualization
                st.subheader("Similarity Score Distribution")
                plt.figure(figsize=(10, 6))
                sns.histplot(filtered_df['similarity_score'], bins=20, kde=True)
                plt.xlabel("Similarity Score")
                plt.ylabel("Frequency")
                plt.title("Distribution of Similarity Scores")
                st.pyplot(plt.gcf())
                
                # Heatmap of Similarity Scores
                st.subheader("Similarity Heatmap")
                similarity_matrix = pd.DataFrame(index=input_df['text'], columns=[row['text'] for _, row in store_df.iterrows()])
                for i, input_row in input_df.iterrows():
                    input_text = input_row['text']
                    for j, store_row in store_df.iterrows():
                        store_text = store_row['text']
                        similarity = cosine_similarity(get_embedding(input_text), store_embeddings[store_text])
                        similarity_matrix.at[input_text, store_text] = similarity

                plt.figure(figsize=(12, 10))
                sns.heatmap(similarity_matrix.astype(float), cmap='YlGnBu', annot=True, fmt='.2f')
                plt.title("Heatmap of Similarity Scores")
                st.pyplot(plt.gcf())

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")

    except ValueError as e:
        st.error(f"Error processing file: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload both store and input Excel files to proceed.")
