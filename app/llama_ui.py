import os
import pickle
import faiss
import torch
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer

# -------------------
# Load Model + Tokenizer
# -------------------
model_name = "NousResearch/Llama-2-7b-chat-hf"
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
# Sentence Transformer for embeddings
# -------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------
# Helper: retrieve top-k chunks
# -------------------
def retrieve(query, k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [chunks[i] for i in I[0] if i != -1]

# -------------------
# Gradio response function
# -------------------
def answer_question(query):
    # Retrieve relevant context
    context_chunks = retrieve(query, k=3)
    if not context_chunks:
        return "I don't know."

    context_text = "\n".join(context_chunks)

    # Build prompt with explicit instructions for detailed answers
    prompt = (
        f"You are a helpful assistant that answers questions about a resume in detail.\n"
        f"Use complete sentences, provide examples, and be thorough.\n\n"
        f"Resume content:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
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
    inputs=gr.Textbox(lines=2, placeholder="Ask me about the resume..."),
    outputs="text",
    title="Resume ChatBot",
    description="Ask questions about the resume and get detailed responses."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
