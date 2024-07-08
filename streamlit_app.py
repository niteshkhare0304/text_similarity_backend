import streamlit as st
import openai
import numpy as np
import torch

st.title("Text Similarity Checker")

# Set your OpenAI API key from environment variable
openai.api_key = st.secrets["OPENAI_API_KEY"]

text1 = st.text_area("Enter first text:")
text2 = st.text_area("Enter second text:")

# Add a slider for similarity threshold
SIMILARITY_THRESHOLD = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.05)

if st.button("Calculate"):
    try:
        # Function to get embeddings from OpenAI
        def get_embedding(text):
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # Change to the desired model
            )
            return np.array(response['data'][0]['embedding'])

        # Calculate embeddings
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            vec1 = torch.tensor(vec1)
            vec2 = torch.tensor(vec2)
            dot_product = torch.dot(vec1, vec2).item()
            magnitude = torch.norm(vec1) * torch.norm(vec2)
            return dot_product / (magnitude + .2)

        similarity = cosine_similarity(embedding1, embedding2)

        # Compare similarity with the threshold
        if similarity > SIMILARITY_THRESHOLD:
            st.write("true")
        else:
            st.write("false")

    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API request failed: {e}")

    except ValueError:
        st.error("Invalid response from OpenAI API")
