import os
import pickle
import faiss
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

# -------------------
# Load Model + Tokenizer
# -------------------
model_name = "NousResearch/Llama-2-7b-chat-hf"  # change if you used another model
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
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

print("Loading FAISS index and chunks...")
index = faiss.read_index(index_file)
with open(chunks_file, "rb") as f:
    chunks = pickle.load(f)

# -------------------
# Helper: retrieve top-k chunks
# -------------------
def retrieve(query, k=3):
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0] if i != -1]

# -------------------
# Main chat loop
# -------------------
print("ChatBot ready! Ask me about my resume (type 'exit' to quit).")

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    # Retrieve relevant context
    context_chunks = retrieve(query, k=3)
    if not context_chunks:
        print("Bot: I don't know.")
        continue

    context_text = "\n".join(context_chunks)

    # Build prompt with context
    prompt = (
        f"You are an assistant answering questions about a resume.\n"
        f"Here is the resume content:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Tokenize & generate
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

    # Strip the prompt back off
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    print(f"Bot: {answer}")
