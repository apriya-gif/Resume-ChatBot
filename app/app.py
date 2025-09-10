# app.py — Privacy-first Flask UI with contact form and PII-guard

from flask import Flask, request, jsonify, render_template_string, send_file
import os, re, json, datetime
import sys
import torch 

# Ensure app package on path for llama imports
sys.path.append('app')

# Import existing RAG + model objects
# - llama_retrieve: top-k text chunks from FAISS built by app/build_index.py
# - model/tokenizer/device: HF transformers model already loaded in llama_ui/llama_query
from llama_ui import llama_retrieve, model, tokenizer, device  # uses existing objects

app = Flask(__name__)

# ---------- Resume parsing (no phone/email in UI) ----------
# Expect data/resume.txt to exist; only derive non-PII fields for display
NAME = "Ameesha Priya"
TITLE = "Software Engineer – Backend, Distributed & Full‑Stack Systems"

# Minimal, resilient parsing so UI can render even if lines shift
def read_resume_lines(path="data/resume.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

lines = read_resume_lines()

# Simple extractors (avoid exposing email/phone)
def get_quick_facts(lines):
    exp = "4+ years"
    edu = "MS Software Engineering, CMU"
    spec = "Distributed Systems"
    recent = "SDE (Capstone/Recent Role)"
    for i, l in enumerate(lines):
        if "years" in l and "experience" in l.lower():
            exp = re.findall(r"\d+\+?\s*years", l)[:1] or [exp]
            exp = exp
        if "Master" in l or "MS Software" in l:
            edu = "MS Software Engineering, CMU"
        if "Distributed" in l:
            spec = "Distributed Systems"
        if "Software Development Engineer" in l:
            recent = "Software Development Engineer (Recent)"
    return exp, edu, spec, recent

EXP, EDU, SPEC, RECENT = get_quick_facts(lines)

CORE_SKILLS = ["Java", "Spring Boot", "Kafka", "AWS", "Kubernetes", "Python"]

# Links (safe to show)
LINKEDIN_URL = "#"
GITHUB_URL = "#"
for l in lines:
    if "linkedin" in l.lower():
        LINKEDIN_URL = ("https://" if not l.startswith("http") else "") + re.search(r"(linkedin\.com\S+)", l).group(1) if re.search(r"(linkedin\.com\S+)", l) else LINKEDIN_URL
    if "github" in l.lower():
        GITHUB_URL = ("https://" if not l.startswith("http") else "") + re.search(r"(github\.com\S+)", l).group(1) if re.search(r"(github\.com\S+)", l) else GITHUB_URL

# ---------- Server-side PII guard ----------
PII_REGEX = re.compile(r"\b(phone|cell|mobile|number|call|text|whats?app|email|e-mail|mail id|contact\s+number)\b", re.I)

POLICY_REFUSAL = (
    "I’m sorry, I cannot share personal phone numbers or private email addresses. "
    "Please use the Contact form on this page to reach out."
)

# ---------- Generation helper ----------
def generate_answer(user_msg: str) -> str:
    # Server-side hard stop for PII requests
    if PII_REGEX.search(user_msg or ""):
        return POLICY_REFUSAL

    chunks = llama_retrieve(user_msg) or []
    context = "\n".join(chunks) if chunks else ""

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
Resume context (scrubbed): 
{context}

Question: {user_msg}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=180,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(outputs, skip_special_tokens=True)
    answer = resp.split("Answer:")[-1].strip()
    # Belt-and-suspenders: sanitize any accidental contact spill
    answer = re.sub(r'\S+@\S+', '[redacted]', answer)
    answer = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[redacted]', answer)
    return answer

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    # Tailwind via CDN; phone/email intentionally omitted
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{NAME} – Resume Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-slate-900 text-slate-100">
  <header class="bg-gradient-to-r from-cyan-600 to-blue-700">
    <div class="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold tracking-tight">{NAME}</h1>
        <p class="text-sm opacity-90">{TITLE}</p>
      </div>
      <div class="flex items-center gap-3">
        <a href="/download" class="px-4 py-2 rounded-md bg-white/10 hover:bg-white/20">Download Resume</a>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Left column -->
    <aside class="lg:col-span-1 space-y-6">
      <section class="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
        <h2 class="font-semibold mb-4">Quick Facts</h2>
        <div class="space-y-3 text-sm">
          <div><span class="text-slate-400">Experience:</span> {EXP}</div>
          <div><span class="text-slate-400">Education:</span> {EDU}</div>
          <div><span class="text-slate-400">Specialization:</span> {SPEC}</div>
          <div><span class="text-slate-400">Recent Role:</span> {RECENT}</div>
        </div>
        <div class="mt-5">
          <h3 class="text-sm font-semibold text-slate-300 mb-2">Core Skills</h3>
          <div class="flex flex-wrap gap-2">
            {''.join([f'<span class="px-2 py-1 text-xs rounded-full bg-slate-700">{s}</span>' for s in CORE_SKILLS])}
          </div>
        </div>
        <div class="mt-5">
          <h3 class="text-sm font-semibold text-slate-300 mb-2">Links</h3>
          <div class="flex flex-col gap-2">
            <a class="text-cyan-300 hover:underline" href="{LINKEDIN_URL}" target="_blank" rel="noopener">LinkedIn</a>
            <a class="text-cyan-300 hover:underline" href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
          </div>
        </div>
      </section>

      <!-- Contact form (no phone; sends to /contact) -->
      <section class="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
        <h2 class="font-semibold mb-4">Contact</h2>
        <form id="contact-form" class="space-y-3">
          <input type="text" name="name" placeholder="Name" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2" required />
          <input type="email" name="email" placeholder="Email" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2" required />
          <input type="text" name="company" placeholder="Company (optional)" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2" />
          <textarea name="message" placeholder="Message" rows="4" class="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2" required></textarea>
          <!-- Honeypot -->
          <input type="text" name="website" class="hidden" tabindex="-1" autocomplete="off" />
          <!-- Consent -->
          <label class="flex items-start gap-2 text-xs text-slate-300">
            <input type="checkbox" name="consent" required class="mt-1" />
            <span>I agree to be contacted and acknowledge the site’s Privacy Policy.</span>
          </label>
          <button type="submit" class="w-full bg-cyan-600 hover:bg-cyan-700 rounded px-3 py-2">Send</button>
          <div id="contact-result" class="text-xs text-slate-300"></div>
        </form>
      </section>
    </aside>

    <!-- Right: Chat -->
    <section class="lg:col-span-2 bg-slate-800/60 border border-slate-700 rounded-xl p-5">
      <h2 class="font-semibold mb-4">Chat with Ameesha</h2>
      <div id="chat" class="h-[520px] overflow-y-auto space-y-3 pr-2">
        <div class="bg-slate-700/50 rounded-lg p-3 text-sm">Hi! I’m an interactive resume assistant. Ask about experience, skills, projects, or education. I will not share phone numbers or private emails; please use the Contact form for outreach.</div>
      </div>
      <div class="mt-4 flex gap-2">
        <input id="msg" class="flex-1 bg-slate-900 border border-slate-700 rounded px-3 py-2" placeholder="Ask about my background..." />
        <button id="send" class="bg-cyan-600 hover:bg-cyan-700 rounded px-4 py-2">Send</button>
      </div>
      <div class="mt-3 flex flex-wrap gap-2">
        <button class="qx px-3 py-1 text-xs rounded bg-slate-700">Tell me about your experience</button>
        <button class="qx px-3 py-1 text-xs rounded bg-slate-700">What projects have you worked on?</button>
        <button class="qx px-3 py-1 text-xs rounded bg-slate-700">What are your technical skills?</button>
        <button class="qx px-3 py-1 text-xs rounded bg-slate-700">Tell me about your education</button>
      </div>
    </section>
  </main>

  <script>
    const chatEl = document.getElementById('chat');
    const msgEl = document.getElementById('msg');
    const sendBtn = document.getElementById('send');

    function addBubble(text, role) {{
      const div = document.createElement('div');
      div.className = (role === 'user')
        ? 'ml-auto max-w-[85%] bg-cyan-700/60 rounded-lg p-3 text-sm'
        : 'max-w-[85%] bg-slate-700/50 rounded-lg p-3 text-sm';
      div.textContent = text;
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
    }}

    async function sendMessage(text) {{
      if (!text) return;
      addBubble(text, 'user');
      msgEl.value = '';
      const r = await fetch('/chat', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ message: text }})
      }});
      const data = await r.json();
      addBubble(data.answer || 'Sorry, something went wrong.', 'assistant');
    }}

    sendBtn.onclick = () => sendMessage(msgEl.value.trim());
    msgEl.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendMessage(msgEl.value.trim());
      }}
    }});

    document.querySelectorAll('.qx').forEach(b => b.onclick = () => sendMessage(b.textContent));

    // Contact form
    const cform = document.getElementById('contact-form');
    const cres = document.getElementById('contact-result');
    cform.addEventListener('submit', async (e) => {{
      e.preventDefault();
      const fd = new FormData(cform);
      if (fd.get('website')) {{  // honeypot
        cres.textContent = 'Submission blocked.';
        return;
      }}
      const payload = Object.fromEntries(fd.entries());
      const r = await fetch('/contact', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      const data = await r.json();
      cres.textContent = data.message || 'Sent.';
      if (data.ok) cform.reset();
    }});
  </script>
</body>
</html>
    """
    return render_template_string(html)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"answer": "Please enter a message."})
    try:
        ans = generate_answer(user_msg)
        return jsonify({"answer": ans})
    except Exception as e:
        return jsonify({"answer": "Sorry, an error occurred generating the response."})

@app.route("/contact", methods=["POST"])
def contact():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    company = (data.get("company") or "").strip()
    message = (data.get("message") or "").strip()
    honeypot = (data.get("website") or "").strip()
    consent = str(data.get("consent") or "").lower() in ("true", "1", "on", "yes")

    if honeypot:
        return jsonify({"ok": False, "message": "Blocked."}), 400
    if not (name and email and message and consent):
        return jsonify({"ok": False, "message": "Please complete all required fields and consent."}), 400

    os.makedirs("data", exist_ok=True)
    row = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "name": name, "email": email, "company": company, "message": message
    }
    with open("data/contact_submissions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    # In production, send email or integrate with a form backend here.
    return jsonify({"ok": True, "message": f"Thank you {name}! Message received."})

@app.route("/download", methods=["GET"])
def download():
    pdf_path = "data/resume.pdf"
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return "Resume PDF not found.", 404

if __name__ == "__main__":
    # flask run style: python app/app.py
    import os
    from pyngrok import ngrok
    authtoken = os.getenv("NGROK_AUTH_TOKEN")
    if not authtoken:
        raise RuntimeError("Please set NGROK_AUTH_TOKEN environment variable before running app.py")
    ngrok.set_auth_token(authtoken)

    # Start Flask app
    port = 7860
    public_url = ngrok.connect(port)
    print(f" * Public URL: {public_url}")

    app.run(host="0.0.0.0", port=port)
