import os
import re
import torch
from flask import Flask, request, render_template_string, send_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= CONFIG =================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + tokenizer
print("Loading model... this may take a few minutes")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

app = Flask(__name__)

# ================= DATA =================
NAME = "Ameesha Priya"
TITLE = "Software Engineer – Backend, Distributed & FullStack Systems"

CORE_SKILLS = [
    "Distributed Systems", "Java", "Go", "Python", "SQL",
    "Microservices", "APIs", "Cloud (GCP, AWS)", "Debugging", "Full-Stack"
]

PII_REGEX = re.compile(r'(\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\S+@\S+)')
POLICY_REFUSAL = "I’m sorry, I cannot share personal phone numbers or private email addresses. Please use the Contact form on this page to reach out."

# ================= HELPERS =================
def llama_retrieve(query: str):
    # Simple stub: real version could use embeddings
    return ["Ameesha has 4+ years of experience building distributed, real-time, and big data systems."]

def generate_answer(user_msg: str) -> str:
    if PII_REGEX.search(user_msg or ""):
        return POLICY_REFUSAL

    chunks = llama_retrieve(user_msg) or []
    context = "\n".join(chunks)

    system_rules = """
You are Ameesha Priya’s professional resume assistant.
STRICT PRIVACY POLICY:
- Never share phone numbers or private email addresses.
- If asked for phone/email, reply exactly: "I’m sorry, I cannot share personal phone numbers or private email addresses. Please use the Contact form on this page to reach out."
- Focus only on professional topics from the provided resume context; do not invent details.
Safe Contact Instruction:
- Direct users to the on-page Contact form for reaching out.
"""

    prompt = f"""{system_rules}
Resume context:
{context}

Question: {user_msg}
Answer:"""

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = resp.split("Answer:")[-1].strip()
    except Exception as e:
        print("Error in generate_answer:", e)
        return "Sorry, an error occurred generating the response."

    # scrub contacts
    answer = re.sub(r'\S+@\S+', '[redacted]', answer)
    answer = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[redacted]', answer)
    return answer

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{{name}} – Interactive Resume</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans">
  <header class="backdrop-blur bg-white/10 shadow-lg sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-extrabold tracking-tight text-white drop-shadow">{{name}}</h1>
        <p class="text-base text-slate-300">{{title}}</p>
      </div>
      <a href="/download" class="px-5 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-90 transition shadow-md">
        Download Resume
      </a>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-3 gap-8">
    <!-- Left column -->
    <aside class="lg:col-span-1 space-y-6">
      <section class="bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-6 shadow-xl transition hover:shadow-2xl">
        <h2 class="font-semibold text-lg mb-4">Quick Facts</h2>
        <ul class="text-sm space-y-2">
          <li>Software Engineer with 4+ years experience</li>
          <li>Focus: Distributed Systems & APIs</li>
          <li>Past domains: Finance, Healthcare, E-commerce</li>
        </ul>
        <div class="flex flex-wrap gap-2 mt-4">
          {% for s in skills %}
          <span class="px-3 py-1 text-xs rounded-full bg-gradient-to-r from-cyan-600 to-blue-700 text-white shadow-md">{{s}}</span>
          {% endfor %}
        </div>
      </section>
      <section class="bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-6 shadow-xl">
        <h2 class="font-semibold text-lg mb-4">Contact</h2>
        <form method="post" action="/contact" class="space-y-3">
          <input name="email" placeholder="Your email" class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"/>
          <textarea name="msg" placeholder="Message" rows="4" class="w-full bg-slate-900/60 border border-slate-700 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"></textarea>
          <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-90 rounded-xl px-3 py-2 shadow-md transition">
            Send
          </button>
        </form>
      </section>
    </aside>

    <!-- Right: Chat -->
    <section class="lg:col-span-2 bg-white/5 backdrop-blur-lg border border-white/10 rounded-2xl p-6 shadow-xl flex flex-col">
      <h2 class="font-semibold text-lg mb-4">Chat with Ameesha</h2>
      <div id="chat" class="flex-1 h-[520px] overflow-y-auto space-y-3 pr-2 bg-slate-900/30 rounded-xl p-4 shadow-inner">
        <div class="bg-slate-700/50 rounded-lg p-3 text-sm animate-fade-in">Hi! I’m an interactive resume assistant. Ask me about my background, skills, or projects!</div>
      </div>
      <div class="mt-4 flex gap-2">
        <input id="msg" class="flex-1 bg-slate-900/60 border border-slate-700 rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500" placeholder="Ask about my background..." />
        <button id="send" class="bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-90 rounded-xl px-5 py-2 shadow-md transition">Send</button>
      </div>
      <div class="mt-3 flex flex-wrap gap-2">
        <button class="qx px-3 py-1 text-xs rounded-xl bg-slate-700/50 hover:bg-slate-600 transition">Tell me about your experience</button>
        <button class="qx px-3 py-1 text-xs rounded-xl bg-slate-700/50 hover:bg-slate-600 transition">What projects have you worked on?</button>
        <button class="qx px-3 py-1 text-xs rounded-xl bg-slate-700/50 hover:bg-slate-600 transition">What are your technical skills?</button>
        <button class="qx px-3 py-1 text-xs rounded-xl bg-slate-700/50 hover:bg-slate-600 transition">Tell me about your education</button>
      </div>
    </section>
  </main>

<script>
const chatBox = document.getElementById("chat");
const sendBtn = document.getElementById("send");
const msgInput = document.getElementById("msg");

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = sender === "user" ? "bg-cyan-600/70 rounded-lg p-3 text-sm ml-auto max-w-[80%]" : "bg-slate-700/50 rounded-lg p-3 text-sm mr-auto max-w-[80%]";
  div.innerText = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;
  appendMessage("user", text);
  msgInput.value = "";

  const res = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({msg: text})
  });
  const data = await res.json();
  appendMessage("bot", data.answer);
}

sendBtn.onclick = sendMessage;
msgInput.addEventListener("keypress", e => { if (e.key === "Enter") sendMessage(); });

document.querySelectorAll(".qx").forEach(b => b.onclick = () => { msgInput.value = b.innerText; sendMessage(); });
</script>
</body>
</html>
    """, name=NAME, title=TITLE, skills=CORE_SKILLS)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    msg = data.get("msg", "")
    ans = generate_answer(msg)
    return {"answer": ans}

@app.route("/download")
def download_resume():
    return send_file("data/resume.pdf", as_attachment=True)

@app.route("/contact", methods=["POST"])
def contact():
    email = request.form.get("email")
    msg = request.form.get("msg")
    print("Contact request from:", email, "Message:", msg)
    return "Thanks! Your message has been sent."
    
if __name__ == "__main__":
    port = 7860
    public_url = ngrok.connect(port)   # ✅ creates tunnel
    print("Public URL:", public_url)
    app.run(host="0.0.0.0", port=port)
