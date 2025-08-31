import os
import pickle
import faiss
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import gradio as gr
from sentence_transformers import SentenceTransformer

# -------------------
# Load Model + Tokenizer
# -------------------
model_name = "NousResearch/Llama-2-7b-chat-hf"  # change if needed
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
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # load once at startup

def retrieve(query, k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0] if i != -1]

# -------------------
# Gradio interface function
# -------------------
def answer_question(query):
    context_chunks = retrieve(query, k=3)
    if not context_chunks:
        return "I don't know."

    context_text = "\n".join(context_chunks)

    prompt = (
        f"You are an assistant answering questions about a resume.\n"
        f"Here is the resume content:\n{context_text}\n\n"
        f"Question: {query}\n"
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
    return answer

# -------------------
# Launch Gradio UI
# -------------------
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Query", placeholder="Ask about the resume..."),
    outputs=gr.Textbox(label="Answer"),
    title="Resume ChatBot",
    description="Ask questions about a resume and get AI-powered answers."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
