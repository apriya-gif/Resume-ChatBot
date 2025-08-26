import faiss
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models
#t5_model = T5ForConditionalGeneration.from_pretrained("t5-large")
#t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "google/flan-t5-xl"  # or "google/flan-t5-xxl"
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load FAISS index and chunks
index = faiss.read_index("models/resume.index")
with open("models/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Retrieve relevant chunks
def retrieve_chunks(query, top_k=2):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)
    return [chunks[i] for i in I[0]]

# Generate answer with T5
def generate_answer(query, context):
    input_text = f"question: {query} context: {context}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = t5_model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Chat loop
if __name__ == "__main__":
    print("Resume Chatbot ready! Ask me anything (type 'exit' to quit).")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        relevant_chunks = retrieve_chunks(query, top_k=3)
        context = " ".join(relevant_chunks)
        answer = generate_answer(query, context)
        print("Bot:", answer)
