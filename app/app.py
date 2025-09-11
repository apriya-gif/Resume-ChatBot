import os
import re
import torch
import requests
from flask import Flask, request, render_template_string, send_file, Response
from transformers import AutoModelForCausalLM, AutoTokenizer


SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")


# ================= CONFIG =================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"  # Force CPU usage for consistent local testing

# Load model + tokenizer
print("Loading lightweight model for local testing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=None,  # Let it use CPU
    low_cpu_mem_usage=True
)
model.to(DEVICE)

app = Flask(__name__)

# ================= DATA =================
NAME = "Ameesha Priya"
TITLE = "Software Engineer – Backend, Distributed & FullStack Systems"

RESUME_CONTENT = """
AMEESHA PRIYA
Software Engineer – Backend, Distributed & FullStack Systems

SUMMARY
Backend-focused Software Engineer with 4+ years architecting distributed systems across finance, healthcare, and e-commerce. Expert in building scalable microservices using Java, Kafka, Spring Boot, and Kubernetes on AWS/GCP/Azure. Proven track record delivering production-grade solutions with quantified business impact.

TECHNICAL SKILLS
Languages: Java, Python, SQL, JavaScript, TypeScript, HTML, CSS, GraphQL
Frameworks & Libraries: Spring Boot, ReactJS, Node.js, JUnit, Mockito
Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, Terraform, Helm, CloudFormation
Data & Streaming: Kafka, Kinesis, Databricks, MongoDB, Redis, Cassandra, Neo4j, Spark, S3

PROFESSIONAL EXPERIENCE
Software Development Engineer, Capstone Project (January 2024 - December 2024) - Sheetz
• Scaled Sheetz's operations from 700 to 1300 stores by developing an event syndicator
• Identified optimal event streaming solution by evaluating Kafka, Kinesis, and Pulsar on AWS
• Increased system scalability by 75% and reduced critical system alerts by 40%

Software Development Engineer (June 2022 - July 2023) - Bank of America
• Delivered automation tools for Merrill Lynch derivative trading, eliminating 100% manual effort weekly
• Improved production batch stability by 85% through automated monitoring systems
• Led technical liaison role between US-India teams, reducing outage resolution time by 45%

Software Development Engineer (July 2021 - June 2022) - Brillio
• Configured and refined API infrastructure by migrating SOAP to REST
• Improved code integration with 85% test coverage using Postman
• Developed robust microservices using Spring Boot + ReactJS

Associate Software Development Engineer (August 2020 - July 2021) - Accenture
• Delivered critical features for AstraZeneca's VeevaCRM solutions
• Managed feature implementations reducing system downtime by 20%

EDUCATION
Master of Software Engineering - Carnegie Mellon University (December 2024)
Bachelor of Computer Science and Engineering - Kalinga Institute of Industrial Technology (July 2020)

AWARDS
• Silver Award - Bank of America (Q1 2023)
• Top 4 – Accenture x Salesforce Hackathon (2021)
"""

CORE_SKILLS = [
    "Java", "Python", "Spring Boot", "Kafka", "Kubernetes", 
    "AWS", "GCP", "Azure", "Microservices", "Distributed Systems"
]

PII_REGEX = re.compile(r'(\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\S+@\S+)')
POLICY_REFUSAL = "I'm sorry, I cannot share personal phone numbers or private email addresses. Please use the Contact form on this page to reach out."

# ================= HELPERS =================
def llama_retrieve(query: str):
    """
    Return relevant resume fragments (as a list of strings) for the given query.
    Kept simple: keyword matching to preserve original behaviour.
    """
    query_lower = (query or "").lower()
    relevant_sections = []
    
    if any(word in query_lower for word in ['experience', 'work', 'job', 'role', 'position']):
        relevant_sections.append(
            "Professional Experience: 4+ years at Bank of America, Brillio, Accenture, and Sheetz. Led automation tools for derivative trading, scaled operations from 700 to 1300 stores, developed microservices handling $50M+ daily volume."
        )
    
    if any(word in query_lower for word in ['skill', 'technical', 'technology', 'programming']):
        relevant_sections.append(
            "Technical Skills: Java, Python, Spring Boot, Kafka, Kubernetes, AWS, GCP, Azure, Docker, ReactJS, MongoDB, Redis. Expert in distributed systems and microservices architecture."
        )
    
    if any(word in query_lower for word in ['project', 'built', 'developed', 'created']):
        relevant_sections.append(
            "Key Projects: Event syndicator for Sheetz scaling to 1300 stores, automation tools for Merrill Lynch trading, SOAP to REST migration at Brillio, real-time data processing with Kafka and Samza."
        )
    
    if any(word in query_lower for word in ['education', 'degree', 'university', 'school']):
        relevant_sections.append(
            "Education: Master of Software Engineering from Carnegie Mellon University (2024), Bachelor of Computer Science from Kalinga Institute of Industrial Technology (2020)."
        )
    
    if any(word in query_lower for word in ['award', 'achievement', 'recognition']):
        relevant_sections.append(
            "Awards: Silver Award from Bank of America (Q1 2023), Top 4 in Accenture x Salesforce Hackathon (2021)."
        )
    
    if not relevant_sections:
        relevant_sections.append(
            "Ameesha Priya is a Backend-focused Software Engineer with 4+ years architecting distributed systems across finance, healthcare, and e-commerce. Expert in Java, Kafka, Spring Boot, and Kubernetes on AWS/GCP/Azure."
        )
    
    return relevant_sections

def _clean_model_output(resp: str, prompt: str) -> str:
    """
    Robustly remove the prompt portion from the model output and return only the assistant's answer.
    Approach:
    1) If model outputs an explicit assistant marker like '<|assistant|>' use that.
    2) Otherwise, remove the first occurrence of the prompt string.
    3) If neither works, fallback to remove everything up to 'Answer:' or last newline.
    """
    if not resp:
        return ""

    # attempt to find explicit assistant marker
    if "<|assistant|>" in resp:
        return resp.split("<|assistant|>")[-1].strip()

    # attempt to remove prompt string if present
    idx = resp.find(prompt)
    if idx != -1:
        return resp[idx + len(prompt):].strip()

    # fallback: try to find "Answer:" or last newline
    # remove everything up to the last occurrence of "Answer:" (case-insensitive)
    lower = resp.lower()
    if "answer:" in lower:
        pos = lower.rfind("answer:")
        return resp[pos + len("answer:"):].strip()

    # final fallback: strip the prompt-like first line(s)
    lines = resp.splitlines()
    # if >1 line, return from second line to end
    if len(lines) > 1:
        return "\n".join(lines[1:]).strip()

    return resp.strip()

def generate_answer(user_msg: str) -> str:
    """Generate a resume-grounded answer and prevent echoing or PII leaks."""
    # PII check on the incoming user message
    if PII_REGEX.search(user_msg or ""):
        return POLICY_REFUSAL

    # Build context from resume fragments (keeps your simple retrieval logic)
    chunks = llama_retrieve(user_msg) or []
    context = "\n".join(chunks)

    # System rules kept strict and concise
    system_rules = (
        "You are Ameesha Priya's resume assistant.Only answer questions using the resume content below. If the answer is not in the resume, say: That information is not available. Resume: ========== {resume_text}==========\n"
        "If asked for private contact details, reply exactly: "
        "'I'm sorry, I cannot share personal phone numbers or private email addresses. Please use the Contact form on this page to reach out.'\n"
        "Keep answers concise, professional, and do not repeat the user's prompt."
    )

    # Use a plain chat-style prompt (no weird token markup needed)
    # NOTE: we intentionally avoid complex role tokens so the causal LM doesn't echo them back verbatim.
    prompt = (
        f"Resume context:\n{context}\n\n"
        f"User question: {user_msg}\n\n"
        f"Answer:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Increase max_new_tokens so the model can complete a full sentence/paragraph.
        # Use do_sample=False to reduce odd echoing (deterministic completion); you can flip to True if you prefer sampling.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,       # bigger to avoid truncation mid-sentence
                temperature=0.0,         # deterministic; set >0 for sample diversity
                top_p=0.9,
                do_sample=False,        # deterministic completion reduces odd repeats
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # strip model echo/prompts robustly
        answer = _clean_model_output(resp, prompt)

    except Exception as e:
        print("Error in generate_answer:", e)
        return "Sorry, an error occurred generating the response."

    # post-process: redact any PII that may still appear
    answer = re.sub(r'\S+@\S+', '[redacted]', answer)
    answer = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[redacted]', answer)
    return answer

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{name}} – Interactive Resume</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .glass { backdrop-filter: blur(20px); background: rgba(255, 255, 255, 0.05); }
    .glass-strong { backdrop-filter: blur(24px); background: rgba(255, 255, 255, 0.08); }
    .animate-fade-in { animation: fadeIn 0.6s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .gradient-text { background: linear-gradient(135deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .shadow-premium { box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08); }
    .border-premium { border: 1px solid rgba(255, 255, 255, 0.1); }
    .hover-lift { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
    .hover-lift:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100 antialiased">
  <!-- Updated header with premium glass morphism design -->
  <header class="glass-strong border-b border-white/10 sticky top-0 z-50 shadow-premium">
    <div class="max-w-6xl mx-auto px-6 py-8 flex items-center justify-between">
      <div class="space-y-1">
        <h1 class="text-4xl font-bold tracking-tight gradient-text">{{name}}</h1>
        <p class="text-slate-400 font-medium">{{title}}</p>
      </div>
      <a href="/download" class="group px-6 py-3 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 transition-all duration-300 shadow-lg hover:shadow-xl font-medium">
        <span class="flex items-center gap-2">
          Download Resume
          <svg class="w-4 h-4 group-hover:translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
          </svg>
        </span>
      </a>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-6 py-12 grid grid-cols-1 lg:grid-cols-5 gap-8">
    <!-- Enhanced left sidebar with premium card design -->
    <aside class="lg:col-span-2 space-y-6">
      <section class="glass border-premium rounded-3xl p-8 shadow-premium hover-lift">
        <h2 class="text-xl font-semibold mb-6 text-white">Quick Facts</h2>
        <div class="space-y-4 text-slate-300">
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-cyan-400"></div>
            <span>4+ years experience</span>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-blue-400"></div>
            <span>Distributed Systems Expert</span>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-indigo-400"></div>
            <span>Full-Stack Engineer</span>
          </div>
        </div>
        
        <div class="mt-8">
          <h3 class="text-sm font-medium text-slate-400 mb-4 uppercase tracking-wider">Core Skills</h3>
          <div class="flex flex-wrap gap-2">
            {% for s in skills %}
            <span class="px-4 py-2 text-sm rounded-xl bg-gradient-to-r from-slate-800 to-slate-700 border border-slate-600 text-slate-200 hover:from-slate-700 hover:to-slate-600 transition-all duration-200">{{s}}</span>
            {% endfor %}
          </div>
        </div>
      </section>

      <!-- Premium contact form with better styling -->
      <section class="glass border-premium rounded-3xl p-8 shadow-premium hover-lift">
        <h2 class="text-xl font-semibold mb-6 text-white">Get In Touch</h2>
        <form method="post" action="/contact" class="space-y-4">
          <div>
            <input name="email" placeholder="Your email address" 
                   class="w-full bg-slate-900/60 border border-slate-700 rounded-2xl px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200"/>
          </div>
          <div>
            <textarea name="msg" placeholder="Your message" rows="4" 
                      class="w-full bg-slate-900/60 border border-slate-700 rounded-2xl px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200 resize-none"></textarea>
          </div>
          <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 rounded-2xl px-4 py-3 font-medium shadow-lg hover:shadow-xl transition-all duration-300">
            Send Message
          </button>
        </form>
      </section>
    </aside>

    <!-- Premium chat interface with enhanced design -->
    <section class="lg:col-span-3 glass border-premium rounded-3xl p-8 shadow-premium hover-lift flex flex-col">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-xl font-semibold text-white">Chat with Ameesha</h2>
        <div class="flex items-center gap-2 text-sm text-slate-400">
          <div class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
          Online
        </div>
      </div>
      
      <div id="chat" class="flex-1 h-[480px] overflow-y-auto space-y-4 pr-2 bg-slate-950/40 rounded-2xl p-6 border border-slate-800/50 shadow-inner">
        <div class="bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl p-4 text-slate-200 animate-fade-in border border-slate-600/30 shadow-lg">
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center text-white font-medium text-sm">A</div>
            <div>
              <p class="font-medium text-sm text-slate-300 mb-1">Ameesha</p>
              <p>Hi! I'm an interactive resume assistant. Ask me about my background, skills, projects, or experience!</p>
            </div>
          </div>
        </div>
      </div>
      
      <div class="mt-6 space-y-4">
        <div class="flex gap-3">
          <input id="msg" class="flex-1 bg-slate-900/60 border border-slate-700 rounded-2xl px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200" 
                 placeholder="Ask about my background, skills, or projects..." />
          <button id="send" class="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 rounded-2xl px-6 py-3 font-medium shadow-lg hover:shadow-xl transition-all duration-300 flex items-center gap-2">
            <span>Send</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
            </svg>
          </button>
        </div>
        
        <div class="flex flex-wrap gap-2">
          <button class="qx px-4 py-2 text-sm rounded-xl bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 hover:text-slate-200 transition-all duration-200">Tell me about your experience</button>
          <button class="qx px-4 py-2 text-sm rounded-xl bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 hover:text-slate-200 transition-all duration-200">What projects have you worked on?</button>
          <button class="qx px-4 py-2 text-sm rounded-xl bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 hover:text-slate-200 transition-all duration-200">What are your technical skills?</button>
          <button class="qx px-4 py-2 text-sm rounded-xl bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700 text-slate-300 hover:text-slate-200 transition-all duration-200">Tell me about your education</button>
        </div>
      </div>
    </section>
  </main>

<script>
const chatBox = document.getElementById("chat");
const sendBtn = document.getElementById("send");
const msgInput = document.getElementById("msg");

function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.className = "animate-fade-in";
  
  if (sender === "user") {
    div.innerHTML = `
      <div class="flex justify-end mb-4">
        <div class="bg-gradient-to-r from-cyan-600 to-blue-700 rounded-2xl p-4 max-w-[80%] shadow-lg">
          <p class="text-white">${text}</p>
        </div>
      </div>
    `;
  } else {
    div.innerHTML = `
      <div class="flex items-start gap-3 mb-4">
        <div class="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center text-white font-medium text-sm flex-shrink-0">A</div>
        <div class="bg-gradient-to-r from-slate-800 to-slate-700 rounded-2xl p-4 max-w-[80%] border border-slate-600/30 shadow-lg">
          <p class="font-medium text-sm text-slate-300 mb-1">Ameesha</p>
          <p class="text-slate-200">${text}</p>
        </div>
      </div>
    `;
  }
  
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;
  
  appendMessage("user", text);
  msgInput.value = "";
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<span>Sending...</span>';

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({msg: text})
    });
    const data = await res.json();
    appendMessage("bot", data.answer);
  } catch (error) {
    appendMessage("bot", "Sorry, there was an error processing your request.");
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = `
      <span>Send</span>
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
      </svg>
    `;
  }
}

sendBtn.onclick = sendMessage;
msgInput.addEventListener("keypress", e => { 
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

document.querySelectorAll(".qx").forEach(b => b.onclick = () => { 
  msgInput.value = b.innerText; 
  sendMessage(); 
});
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
    pdf_paths = [
        "resume.pdf"
    ]
    
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=True, download_name="Ameesha_Priya_Resume.pdf")
    
    return Response(
        RESUME_CONTENT,
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment; filename=Ameesha_Priya_Resume.txt'}
    )

@app.route("/contact", methods=["POST"])
def contact():
    email = request.form.get("email")
    msg = request.form.get("msg")
    if email and msg:
        print("Contact request from:", email, "Message:", msg)
        send_email("Contact Form", email, msg)
        return "Thanks! Your message has been sent."
    return "Missing email or message", 400

def send_email(name, email, message):
    if not SENDGRID_API_KEY:
        print("SENDGRID_API_KEY not set. Cannot send email.")
        return
    data = {
        "personalizations": [{"to": [{"email": "apriya.gcp@gmail.com"}]}],
        "from": {"email": "no-reply@yourdomain.com"},
        "subject": f"New message from {name}",
        "content": [{"type": "text/plain", "value": f"From: {email}\n\n{message}"}],
    }
    response = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
        json=data,
    )
    if response.status_code >= 400:
        print("SendGrid email error:", response.text)
    
if __name__ == "__main__":
    port = 7860
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
