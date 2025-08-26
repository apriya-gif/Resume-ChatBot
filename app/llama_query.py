# app/llama_query.py
import os, re, pickle, numpy as np, faiss, torch
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------- Config ----------------------------
MODEL_ID = "tiiuae/falcon-7b-instruct"
   # chat-tuned, open
# Alternative: MODEL_ID = "tiiuae/falcon-7b-instruct"

TOP_K = 5                 # retrieve this many sentences
CTX_TOKEN_BUDGET = 900    # pack retrieved context up to ~900 tokens (safe on 4k ctx)
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3         # lower = more factual
TOP_P = 0.95
REPETITION_PENALTY = 1.1
IDK_THRESHOLD = 0.25      # cosine sim threshold; if best < this -> "I don't know"

MODELS_DIR = "models"
INDEX_PATH = os.path.join(MODELS_DIR, "resume.index")
CHUNKS_PATH = os.path.join(MODELS_DIR, "chunks.pkl")

# ------------------------ Load retriever ------------------------
if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
    raise FileNotFoundError(
        "Missing FAISS index or chunks. Run: python app/build_index.py"
    )

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    CHUNKS: List[str] = pickle.load(f)

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------- Load chat LLM (4-bit) --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    print("⚠️ No GPU detected. Generations will be slow and less accurate.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto" if device == "cuda" else None,
    quantization_config=bnb_config if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.eval()

# ------------------------ Helper funcs --------------------------
def clean_context(text: str) -> str:
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'linkedin\.com\S+', '', text)
    text = re.sub(r'github\.com\S+', '', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    return text.strip()

def search(query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    q = EMBEDDER.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(q, dtype="float32"), top_k)
    # D are cosine sims because index is IP on normalized vectors
    return [(CHUNKS[i], float(D[0][j])) for j, i in enumerate(I[0])]

def pack_context(chunks_and_scores: List[Tuple[str, float]], token_budget: int) -> str:
    ctx_parts = []
    used = 0
    for chunk, _ in chunks_and_scores:
        t = tokenizer(chunk, add_special_tokens=False, return_tensors="pt")
        length = t.input_ids.shape[1]
        if used + length > token_budget:
            break
        ctx_parts.append(f"- {chunk}")
        used += length
    return "\n".join(ctx_parts)

def make_prompt(query: str, context: str, history: List[Tuple[str, str]]) -> str:
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-3:]])
    system = (
        "You are a helpful assistant for answering questions about Ameesha Priya's resume. "
        "Only use the provided context. If the answer is not present, say exactly: \"I don't know.\" "
        "Be concise (2-4 sentences), conversational, and avoid copying phrases verbatim from context."
    )
    user = (
        f"Chat history (last 3 turns):\n{history_text}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )

    # Prefer chat template if the tokenizer provides one
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Generic fallback instruction style prompt
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

def generate(query: str, history: List[Tuple[str, str]]) -> str:
    hits = search(query, TOP_K)
    best_sim = hits[0][1] if hits else 0.0
    if best_sim < IDK_THRESHOLD:
        return "I don't know."

    context = pack_context(hits, CTX_TOKEN_BUDGET)
    prompt = make_prompt(query, context, history)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                    # deterministic for factual answers
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # remove the prompt portion (causal models echo input)
    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # very short/empty → fallback
    if not text or text.lower() in {"", "i don't know"}:
        return "I don't know."
    # trim any trailing boilerplate
    return re.sub(r"\s+", " ", text).strip()

# -------------------------- CLI loop ----------------------------
def main():
    print("Resume Chatbot ready! Type 'exit' or 'quit' to end.")
    history: List[Tuple[str, str]] = []
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            a = generate(q, history)
        except Exception as e:
            a = f"(Error while generating: {e})"
        print("Bot:", a)
        history.append((q, a))

if __name__ == "__main__":
    main()
