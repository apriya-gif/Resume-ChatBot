# app/api.py
import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# -------------------
# Load model + tokenizer
# -------------------
model_name = "NousResearch/Llama-2-7b-chat-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
model.eval()

# -------------------
# Load FAISS index + chunks
# -------------------
index_file = "models/resume.index"
chunks_file = "models/chunks.pkl"

index = faiss.read_index(index_file)
with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)

# -------------------
# Setup FastAPI
# -------------------
app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    k: int = 3

def retrieve(query, k=3):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0] if i != -1]

@app.post("/query")
def answer_question(req: QueryRequest):
    context_chunks = retrieve(req.question, req.k)
    context_text = "\n".join(context_chunks)
    
    prompt = (
        f"You are an assistant answering questions about a resume.\n"
        f"Here is the resume content:\n{context_text}\n\n"
        f"Question: {req.question}\n"
        f"Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return {"answer": answer}
