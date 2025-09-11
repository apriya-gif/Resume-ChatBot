import os
import re
import torch
from flask import Flask, request, render_template_string, send_file, Response
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= CONFIG =================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"  # Force CPU usage for local testing

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
    query_lower = query.lower()
    relevant_sections = []
    
    if any(word in query_lower for word in ['experience', 'work', 'job', 'role', 'position']):
        relevant_sections.append("Professional Experience: 4+ years at Bank of America, Brillio, Accenture, and Sheetz. Led automation tools for derivative trading, scaled operations from 700 to 1300 stores, developed microservices handling $50M+ daily volume.")
    
    if any(word in query_lower for word in ['skill', 'technical', 'technology', 'programming']):
        relevant_sections.append("Technical Skills: Java, Python, Spring Boot, Kafka, Kubernetes, AWS, GCP, Azure, Docker, ReactJS, MongoDB, Redis. Expert in distributed systems and microservices architecture.")
    
    if any(word in query_lower for word in ['project', 'built', 'developed', 'created']):
        relevant_sections.append("Key Projects: Event syndicator for Sheetz scaling to 1300 stores, automation tools for Merrill Lynch trading, SOAP to REST migration at Brillio, real-time data processing with Kafka and Samza.")
    
    if any(word in query_lower for word in ['education', 'degree', 'university', 'school']):
        relevant_sections.append("Education: Master of Software Engineering from Carnegie Mellon University (2024), Bachelor of Computer Science from Kalinga Institute of Industrial Technology (2020).")
    
    if any(word in query_lower for word in ['award', 'achievement', 'recognition']):
        relevant_sections.append("Awards: Silver Award from Bank of America (Q1 2023), Top 4 in Accenture x Salesforce Hackathon (2021).")
    
    if not relevant_sections:
        relevant_sections.append("Ameesha Priya is a Backend-focused Software Engineer with 4+ years architecting distributed systems across finance, healthcare, and e-commerce. Expert in Java, Kafka, Spring Boot, and Kubernetes on AWS/GCP/Azure.")
    
    return relevant_sections

def generate_answer(user_msg: str) -> str:
    if PII_REGEX.search(user_msg or ""):
        return POLICY_REFUSAL

    chunks = llama_retrieve(user_msg) or []
    context = "\n".join(chunks)

    system_rules = """You are Ameesha Priya's professional resume assistant.
STRICT PRIVACY POLICY:
- Never share phone numbers or private email addresses.
- If asked for phone/email, reply exactly: "I'm sorry, I cannot share personal phone numbers or private email addresses. Please use the Contact form on this page to reach out."
- Focus only on professional topics from the provided resume context; do not invent details.
Safe Contact Instruction:
- Direct users to the on-page Contact form for reaching out."""

    prompt = f"<|system|>\n{system_rules}\n\nResume context:\n{context}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in resp:
            answer = resp.split("<|assistant|>")[-1].strip()
        else:
            answer = resp[len(prompt):].strip()
    except Exception as e:
        print("Error in generate_answer:", e)
        return "Sorry, an error occurred generating the response."

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
<!-- HTML content continues exactly as in your original code -->
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
        "Resume-ChatBot-main/data/resume.pdf",
        "data/resume.pdf", 
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
    print("Contact request from:", email, "Message:", msg)
    return "Thanks! Your message has been sent."
    
if __name__ == "__main__":
    port = 7860
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
