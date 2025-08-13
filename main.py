from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pandas as pd

# Load your CSV file
df = pd.read_csv("../data/cleaning/merged_category_dataset.csv")
texts = df["isbn13_description"].tolist()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define your query
query = input("Enter your query: ")

# Encode the query
query_vector = model.encode([query], convert_to_numpy=True)

# Search in the FAISS index
index = faiss.read_index("../data/vector_embeddings/faiss_index.index")
distances, indices = index.search(query_vector, k=5)

# Print results
print(f"Top matches for query: {query}")
for i, idx in enumerate(indices[0]):
  
    isbn13 = texts[idx].split(" ", 1)[0]
    # Match in original DataFrame
    book_info = df[df["isbn13"].astype(str) == isbn13]
    
    row = book_info.iloc[0]  # first matching row

    title = row.get("full_title", "Unknown Title")
    category = row.get("predicted_category", "Unknown Category")
    description = row.get("description", "No description")

    print(f"{title}")
    print(f"Category: {category}")
    print(f"Description: {description}")
    print("-" * 50)
