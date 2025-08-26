import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------- Resume Text ----------------------------
full_text =  '''AMEESHA PRIYA Software Engineer – Backend, Distributed & FullStack Systems apriya.gcp@gmail.com | (412) 499-6900 | linkedin.com/in/ameesha-priya-2a773a136 | github.com/apriya-gif SUMMARY Backend-focused Software Engineer with 4+ years architecting distributed systems across finance, healthcare, and e-commerce. Expert in building scalable microservices using Java, Kafka, Spring Boot, and Kubernetes on AWS/GCP/Azure. Proven track record delivering production-grade solutions with quantified business impact. TECHNICAL SKILLS Languages: Java, Python, SQL, JavaScript, TypeScript, HTML, CSS, GraphQL Frameworks & Libraries: Spring Boot, ReactJS, Node.js, JUnit, Mockito Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, Terraform, Helm, CloudFormation Data & Streaming: Kafka, Kinesis, Databricks, MongoDB, Redis, Cassandra, Neo4j, Spark, S3 Tools & Infrastructure: gRPC, Git, Postman, JIRA, VSCode, IntelliJ, Eclipse, SonarLint, Nginx, Jupyter PROFESSIONAL EXPERIENCE Software Development Engineer, Capstone Project January 2024 - December 2024 Sheetz (via Carnegie Mellon University) ● Scaled Sheetz's operations from 700 to 1300 stores by developing an event syndicator to streamline data flow into Databricks, enabling 85% faster data processing across retail locations. ● Identified optimal event streaming solution by evaluating Kafka, Kinesis, and Pulsar on AWS using Nginx & Apache JMeter, processing 500,000+ events/second and selecting Kafka for 40% superior performance. ● Increased system scalability by 75% and reduced critical system alerts by 40% using SolarWinds monitoring, ensuring seamless expansion capacity for 1.5x future growth. Software Development Engineer June 2022 - July 2023 Bank of America ● Delivered automation tools for Merrill Lynch derivative trading, eliminating 100% manual effort weekly and reducing SLA breaches by 60%. ● Improved production batch stability by 85% through automated monitoring systems, handling $50M+ daily trading volume. ● Led technical liaison role between US-India teams, reducing outage resolution time by 45% and supervised 3 engineers. ● Architected scalable microservices for counterparty risk management, processing 10K+ transactions daily. Software Development Engineer July 2021 - June 2022 Brillio ● Configured and refined API infrastructure by migrating SOAP based application to REST and adding JDBC for database-application connection. ● Improved code integration, and conducted extensive unit tests and endpoint testing using Postman achieving 85% coverage. ● Developed robust microservices using Spring Boot+ReactJS (backend+frontend framework). ● Enabled seamless data integration and migration in Verizon’s 5G domain by developing APIs, establishing a multi-source data pipeline through automation scripts and comprehensive documentation (LLD, HLD, flow diagrams). Associate Software Development Engineer August 2020 - July 2021 Accenture ● Delivered critical features and stability for AstraZeneca’s VeevaCRM solutions as the key developer for iPatient, managing feature implementations and bug fixes under tight deadlines reducing system downtime by 20%. ACADEMIC PROJECTS Stream Processing with Kafka and Samza - Developed and analyzed a real-time data processing system by using Apache Kafka as the messaging system to handle large streams of incoming data and Apache Samza for processing these streams with minimal delay on AWS. Containers: Docker and Kubernetes - Containerized and orchestrated microservices using Docker and Kubernetes on GCP and Azure, enabling scalable and reliable application deployment and management. Machine Learning on the Cloud - Implemented and deployed an end-to-end machine learning pipeline on GCP, enhancing model accuracy through feature engineering and hyperparameter tuning. EDUCATION School of Computer Science, Carnegie Mellon University December 2024 Master of Software Engineering (Courses: Cloud Computing, Software Architecture, WebApp / TA: Engineering Data Intensive Scalable Systems,QA) Kalinga Institute of Industrial Technology July 2020 Bachelor of Computer Science and Engineering AWARDS AND ACHIEVEMENTS ● Silver Award - Bank of America (Q! 2023) ● Top 4 – Accenture x Salesforce Hackathon (2021)'''

# ---------------------------- Utilities ----------------------------
def clean_context(text):
    """Remove emails, phone numbers, and links to prevent copy-paste answers."""
    text = re.sub(r'\S+@\S+', '', text)         # emails
    text = re.sub(r'http\S+', '', text)         # links
    text = re.sub(r'linkedin\.com\S+', '', text)
    text = re.sub(r'github\.com\S+', '', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    return text.strip()

def split_text_by_sentences(text):
    """Split text into sentences using simple heuristic."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------------------------- Device ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- FAISS + Embeddings ----------------------------
chunks = split_text_by_sentences(clean_context(full_text))
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))

def retrieve_top_sentences(query, top_k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [chunks[i] for i in I[0]]

# ---------------------------- TinyLlama Model ----------------------------
model_name = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------- Answer Generation ----------------------------
def generate_answer(query, top_k=3):
    relevant_chunks = retrieve_top_sentences(query, top_k=top_k)
    merged_context = "\n".join(relevant_chunks)

    prompt = f"""
You are a helpful resume assistant. 
ONLY answer based on CONTEXT. 
If context does not contain the answer, respond with: "I don't know."

Question: {query}
CONTEXT:
{merged_context}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    answer = tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True).strip()

    return answer or "I don't know"



# ---------------------------- Chat Loop ----------------------------
def chat():
    print("Resume Chatbot ready! Type 'exit' or 'quit' to end.")
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = generate_answer(query, top_k=3)
        print("Bot:", answer)
        chat_history.append((query, answer))

if __name__ == "__main__":
    chat()
