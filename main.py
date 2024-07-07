# text_similarity_project/app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel

app = FastAPI()

class Texts(BaseModel):
    text1: str
    text2: str

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Adjusted similarity threshold
SIMILARITY_THRESHOLD = 0.8  # Adjust this threshold as needed

@app.post("/compare_texts")
def compare_texts(texts: Texts) -> bool:
    try:
        # Tokenize input texts
        inputs1 = tokenizer(texts.text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(texts.text2, return_tensors="pt", padding=True, truncation=True)

        # Print tokenized inputs for debugging
        print("Tokenized Inputs:")
        print(inputs1)
        print(inputs2)

        # Ensure token IDs are different
        if torch.equal(inputs1["input_ids"], inputs2["input_ids"]):
            raise HTTPException(status_code=400, detail="Input texts result in identical token IDs.")

        # Encode input texts
        with torch.no_grad():
            embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
            embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity
        similarity = cosine_similarity(embeddings1, embeddings2)

        # Print similarity score for debugging
        print("Similarity Score:", similarity.item())

        # Return similarity result based on adjusted threshold
        return similarity.item() > SIMILARITY_THRESHOLD
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {e}")

def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1.squeeze(), vec2.squeeze())
    magnitude = torch.norm(vec1) * torch.norm(vec2)
    return dot_product / magnitude if magnitude != 0 else 0
