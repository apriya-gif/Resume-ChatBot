import faiss
import numpy as np
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
from sentence_transformers import SentenceTransformer

# Load FAISS index and chunks
index = faiss.read_index("models/resume.index")
with open("models/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a CPU-friendly text generation model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def query_resume(query, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    context = " ".join([chunks[i] for i in indices[0]])
    context = context[:1000]
    prompt = f"""
You are a helpful assistant answering questions about Ameesha Priya's resume.
Only use the context below to answer. 
Summarize clearly and concisely in 3–4 sentences. 
If the answer is not in the context, reply: "I don't know."

Context:
{context}

Question: {query}
"""

    response = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    return response




if __name__ == "__main__":
    query = "Tell me about my projects"
    answer = query_resume(query)
    print("Answer:\n", answer)
