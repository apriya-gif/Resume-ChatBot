# app.py - Flask application for Resume ChatBot UI using Tailwind CSS

from flask import Flask, render_template_string, request, jsonify, send_file
import sys

# Add app directory to path for llama imports (adjust path if needed)
sys.path.append('app')
from llama_ui import llama_retrieve, model, tokenizer, device
import torch

app = Flask(__name__)

# Parse resume data for dynamic content
with open('data/resume.txt', 'r') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

# Extract basic info
name = lines[0]
title = lines[1]
contacts = [p.strip() for p in lines[2].split('|')]
email = contacts[0]
phone = contacts[1]
linkedin_url = 'https://' + contacts[2] if contacts[2].startswith('linkedin') else contacts[2]
github_url = 'https://' + contacts[3] if contacts[3].startswith('github') else contacts[3]

# Extract quick facts
experience = ""
for line in lines:
    if 'years' in line:
        experience = line.split('years')[0].strip() + ' years'
        break
if not experience:
    experience = "N/A"

education = ""
for i, line in enumerate(lines):
    if line.startswith("Master of Software Engineering"):
        uni_line = lines[i-1] if i-1 >= 0 else ""
        education = f"M.S. Software Engineering, {uni_line.split(',')[0]}"
        break
if not education:
    education = "N/A"

specialization = ""
if "Distributed" in title:
    specialization = "Distributed Systems"

recent_role = ""
for i, line in enumerate(lines):
    if line.startswith("Software Development Engineer, Capstone"):
        comp = lines[i+1] if i+1 < len(lines) else ""
        if comp.endswith("University"):
            comp = comp.split(" (")[0]
        recent_role = f"Software Development Engineer at {comp}"
        break
if not recent_role:
    recent_role = "Software Development Engineer"

# Core skills (example list)
core_skills = ["Java", "Spring Boot", "Kafka", "AWS", "Kubernetes", "Python"]

# Links
links = [
    {"name": "LinkedIn", "url": linkedin_url, "icon": "fab fa-linkedin"},
    {"name": "GitHub", "url": github_url, "icon": "fab fa-github"},
]

@app.route("/")
def index():
    # HTML template with Tailwind CSS for styling
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chat with Ameesha</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-900 text-white min-h-screen">

    <!-- Header -->
    <header class="bg-gradient-to-r from-teal-600 to-blue-600 p-6 flex items-center justify-between">
        <div>
            <h1 class="text-3xl font-bold">{name}</h1>
            <p class="text-xl">{title}</p>
            <div class="mt-2 text-sm">
                <span class="mr-4"><i class="fas fa-envelope"></i> {email}</span>
                <span><i class="fas fa-phone"></i> {phone}</span>
            </div>
        </div>
        <div>
            <a href="/download" class="bg-blue-800 hover:bg-blue-700 text-white px-4 py-2 rounded">Download Resume</a>
        </div>
    </header>

    <div class="flex flex-col md:flex-row flex-1">
        <!-- Sidebar -->
        <aside class="w-full md:w-1/3 lg:w-1/4 bg-gray-800 p-6">
            <h2 class="text-xl font-semibold mb-4">Quick Facts</h2>
            <div class="space-y-2">
                <div class="flex justify-between text-gray-300">
                    <span class="font-medium">Experience</span>
                    <span>{experience}</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span class="font-medium">Education</span>
                    <span>{education}</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span class="font-medium">Specialization</span>
                    <span>{specialization}</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span class="font-medium">Recent Role</span>
                    <span>{recent_role}</span>
                </div>
            </div>

            <h2 class="text-xl font-semibold mt-6 mb-4">Core Skills</h2>
            <div class="flex flex-wrap">
'''
    # Add skill tags
    for skill in core_skills:
        html += f'                <span class="bg-teal-700 text-white rounded-full px-3 py-1 mr-2 mb-2">{skill}</span>\n'
    html += '''
            </div>

            <h2 class="text-xl font-semibold mt-6 mb-4">Links</h2>
            <div class="space-y-2">
'''
    for link in links:
        html += f'                <a href="{link["url"]}" target="_blank" class="flex items-center text-teal-200 hover:text-white"><i class="{link["icon"]} fa-lg mr-2"></i> {link["name"]}</a>\n'
    html += '''
            </div>
        </aside>

        <!-- Chat section -->
        <main class="flex-1 p-6 flex flex-col">
            <h2 class="text-2xl font-semibold mb-2">Chat with Ameesha</h2>
            <p class="mb-4 text-gray-400">Ask me anything about my experience, skills, or projects!</p>
            <div id="chat-area" class="flex-1 overflow-y-auto bg-gray-800 p-4 rounded-lg mb-4">
                <!-- Assistant greeting message -->
                <div class="flex items-start mb-4">
                    <div class="bg-teal-600 text-white font-bold rounded-full flex items-center justify-center" style="width:40px; height:40px;">AP</div>
                    <div class="bg-gray-700 text-white p-3 ml-2 rounded-lg max-w-prose">
                        Hi! I'm Ameesha Priya's interactive resume assistant. I can tell you about my experience, skills, projects, education, and more. Feel free to ask me anything!
                    </div>
                </div>
            </div>

            <!-- Suggested prompts -->
            <div class="mb-4 space-x-2">
                <button onclick="setQuestion('Tell me about your experience')" class="bg-blue-700 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm">Tell me about your experience</button>
                <button onclick="setQuestion('What projects have you worked on?')" class="bg-blue-700 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm">What projects have you worked on?</button>
                <button onclick="setQuestion('What are your technical skills?')" class="bg-blue-700 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm">What are your technical skills?</button>
                <button onclick="setQuestion('Tell me about your education')" class="bg-blue-700 hover:bg-blue-600 text-white px-3 py-1 rounded text-sm">Tell me about your education</button>
            </div>

            <!-- Input area -->
            <div class="flex">
                <input id="chat-input" type="text" placeholder="Ask me anything about my background..." class="flex-1 bg-gray-700 text-white px-3 py-2 rounded-l focus:outline-none" onkeydown="if(event.key === 'Enter') sendMessage();">
                <button onclick="sendMessage()" class="bg-blue-600 hover:bg-blue-500 px-4 rounded-r">
                    <i class="fas fa-paper-plane text-white"></i>
                </button>
            </div>
        </main>
    </div>

    <!-- Chat JS -->
    <script>
        function addMessage(role, text) {
            const chatArea = document.getElementById('chat-area');
            const wrapper = document.createElement('div');
            wrapper.className = 'flex mb-4 ' + (role === 'user' ? 'justify-end' : 'justify-start');
            const contentDiv = document.createElement('div');
            contentDiv.className = (role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-white') + ' p-3 rounded-lg max-w-prose';
            contentDiv.innerText = text;
            if (role === 'assistant') {
                const avatar = document.createElement('div');
                avatar.className = 'bg-teal-600 text-white font-bold rounded-full flex items-center justify-center mr-2';
                avatar.style.width = avatar.style.height = '40px';
                avatar.innerText = 'AP';
                wrapper.appendChild(avatar);
                wrapper.appendChild(contentDiv);
            } else {
                wrapper.appendChild(contentDiv);
            }
            chatArea.appendChild(wrapper);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('chat-input');
            const question = input.value.trim();
            if (!question) return;
            addMessage('user', question);
            input.value = '';
            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
            .then(data => {
                addMessage('assistant', data.answer);
            });
        }

        function setQuestion(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }
    </script>
</body>
</html>
'''
    return html

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please ask a question."})
    chunks = llama_retrieve(question)
    context = "\\n".join(chunks)
    prompt = f\"\"\"You are Ameesha Priya's AI assistant. Answer questions about her resume and background.
IMPORTANT: Do not share personal contact info. If asked, refer to contact form.
Context:\\n{context}\\nQuestion: {question}\\nAnswer:\"\"\"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, max_new_tokens=150,
            temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[-1].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    return jsonify({"answer": answer})

@app.route("/download")
def download():
    return send_file("data/resume.txt", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
