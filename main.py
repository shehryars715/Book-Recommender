import os
import pickle
import faiss
from huggingface_hub import InferenceClient
import numpy as np

# 1️⃣ Load FAISS index and metadata
index = faiss.read_index("vectors.faiss")
with open("metadata.pkl", "rb") as f:
    texts = pickle.load(f)

# 2️⃣ Setup Hugging Face API client
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

# 3️⃣ Define your query
query = "AI in education"

# 4️⃣ Get embedding from Hugging Face API (feature extraction)
embedding = client.feature_extraction(
    model="sentence-transformers/all-MiniLM-L6-v2",
    inputs=query
)

# Convert to NumPy array (FAISS requires float32)
query_vector = np.array(embedding, dtype="float32").reshape(1, -1)

# 5️⃣ Search in FAISS
distances, indices = index.search(query_vector, k=5)

# 6️⃣ Show results
print("Query:", query)
print("\nTop matches:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"Text: {texts[idx]}  |  Distance: {dist:.4f}")
