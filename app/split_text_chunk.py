from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load your resume text
full_text =  '''Resume Text'''
def split_text(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

chunks = split_text(full_text)
print(f"Created {len(chunks)} chunks")

# Load model and encode
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")
embeddings = model.encode(chunks)
print("Encoding done!")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print("FAISS index created!")

# Save index and chunks for later use
faiss.write_index(index, "models/resume.index")
with open("models/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Index and chunks saved to models/")
