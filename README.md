# Resume-ChatBot

This project lets you query your resume using embeddings and LLaMA2 for natural language answers.

## Setup

1. Clone the repo:
https://github.com/YOUR_USERNAME/resume-chatbot.git
2. Navigate to app:
cd resume-chatbot/app
3. Install dependencies:
pip install -r requirements.txt
4. Split your resume and create embeddings:
python split_text_chunk.py
5. Query your resume:
python llama_query.py
## Deployment

You can containerize the project using Docker:
docker build -t resume-chatbot .
docker run -it resume-chatbot
## Notes

- Requires Ollama with LLaMA2 running locally.
- FAISS index is stored in `models/resume.index`.