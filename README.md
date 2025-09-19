# Resume-ChatBot 🤖📄  

A conversational chatbot that lets you query my resume in natural language. It uses embeddings + LLaMA2 locally to generate answers, demonstrating skills in **NLP**, **vector search**, and **end-to-end system design**.  

---

## 🧾 Project Overview  

This repository contains the **local version** of the Resume-ChatBot before migrating to AWS Lambda + Bedrock.  
It works by:  
1. Splitting resume text into chunks.  
2. Creating embeddings for each chunk.  
3. Storing them in a vector database (FAISS).  
4. Querying with LLaMA2 to return context-aware answers.  

---

## 📁 Repository Structure  

Resume-ChatBot/
├── app/ # main application code
│ ├── llama_query.py # query resume with LLaMA2 + embeddings
│ ├── split_text_chunk.py # split resume into smaller pieces
│ └── ...
├── data/ # resume and raw data files
├── models/ # embeddings and FAISS index
│ └── resume.index
├── Dockerfile # containerization setup
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md # this file

## 🔧 Setup & Run Locally  

### Prerequisites  
- Python 3.x  
- Git  
- FAISS + Hugging Face (installed via `requirements.txt`)  
- LLaMA2 model available locally (e.g., via Ollama or Hugging Face)  
- (Optional) Docker for containerized setup  

### Installation  

1. Clone the repo:  
   ```bash
   git clone https://github.com/apriya-gif/Resume-ChatBot.git
   cd Resume-ChatBot/app
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3. Prepare resume data & embeddings:
   ```bash
   # Split resume text into chunks
   python split_text_chunk.py
4. Run a query:
   ```bash
   # Build embeddings + FAISS index (saved under models/)
   python llama_query.py

🐳 Run with Docker (Optional)
```bash
docker build -t resume-chatbot .
docker run -it resume-chatbot
```
⚠️ Notes
- This version runs entirely locally with LLaMA2 + FAISS.
- Each time the resume changes, embeddings should be regenerated.
- The FAISS index is stored under models/ (e.g., resume.index).
