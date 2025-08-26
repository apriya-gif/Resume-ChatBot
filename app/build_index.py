# app/build_index.py
import os, re, pickle, numpy as np, faiss
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/resume.txt"   # put your resume text here (plain .txt)
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_context(text):
    text = re.sub(r'\S+@\S+', '', text)         # emails
    text = re.sub(r'http\S+', '', text)         # links
    text = re.sub(r'linkedin\.com\S+', '', text)
    text = re.sub(r'github\.com\S+', '', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    return text.strip()

def split_sentences(text):
    # simple sentence splitter; enough for resumes
    parts = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    return [p.strip() for p in parts if p.strip()]

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Create it and paste your resume text."
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_sentences(clean_context(full_text))

    # embed with cosine-normalized vectors
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # cosine => inner product on normalized vectors
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(OUT_DIR, "resume.index"))
    with open(os.path.join(OUT_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Built index with {len(chunks)} chunks → {OUT_DIR}/resume.index")
