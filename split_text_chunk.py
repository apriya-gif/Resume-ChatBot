from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

full_text = '''AMEESHA PRIYA Software Engineer – Backend, Distributed & FullStack Systems apriya.gcp@gmail.com | (412) 499-6900 | linkedin.com/in/ameesha-priya-2a773a136 | github.com/apriya-gif SUMMARY Backend-focused Software Engineer with 4+ years architecting distributed systems across finance, healthcare, and e-commerce. Expert in building scalable microservices using Java, Kafka, Spring Boot, and Kubernetes on AWS/GCP/Azure. Proven track record delivering production-grade solutions with quantified business impact. TECHNICAL SKILLS Languages: Java, Python, SQL, JavaScript, TypeScript, HTML, CSS, GraphQL Frameworks & Libraries: Spring Boot, ReactJS, Node.js, JUnit, Mockito Cloud & DevOps: AWS, GCP, Azure, Docker, Kubernetes, Terraform, Helm, CloudFormation Data & Streaming: Kafka, Kinesis, Databricks, MongoDB, Redis, Cassandra, Neo4j, Spark, S3 Tools & Infrastructure: gRPC, Git, Postman, JIRA, VSCode, IntelliJ, Eclipse, SonarLint, Nginx, Jupyter PROFESSIONAL EXPERIENCE Software Development Engineer, Capstone Project January 2024 - December 2024 Sheetz (via Carnegie Mellon University) ● Scaled Sheetz's operations from 700 to 1300 stores by developing an event syndicator to streamline data flow into Databricks, enabling 85% faster data processing across retail locations. ● Identified optimal event streaming solution by evaluating Kafka, Kinesis, and Pulsar on AWS using Nginx & Apache JMeter, processing 500,000+ events/second and selecting Kafka for 40% superior performance. ● Increased system scalability by 75% and reduced critical system alerts by 40% using SolarWinds monitoring, ensuring seamless expansion capacity for 1.5x future growth. Software Development Engineer June 2022 - July 2023 Bank of America ● Delivered automation tools for Merrill Lynch derivative trading, eliminating 100% manual effort weekly and reducing SLA breaches by 60%. ● Improved production batch stability by 85% through automated monitoring systems, handling $50M+ daily trading volume. ● Led technical liaison role between US-India teams, reducing outage resolution time by 45% and supervised 3 engineers. ● Architected scalable microservices for counterparty risk management, processing 10K+ transactions daily. Software Development Engineer July 2021 - June 2022 Brillio ● Configured and refined API infrastructure by migrating SOAP based application to REST and adding JDBC for database-application connection. ● Improved code integration, and conducted extensive unit tests and endpoint testing using Postman achieving 85% coverage. ● Developed robust microservices using Spring Boot+ReactJS (backend+frontend framework). ● Enabled seamless data integration and migration in Verizon’s 5G domain by developing APIs, establishing a multi-source data pipeline through automation scripts and comprehensive documentation (LLD, HLD, flow diagrams). Associate Software Development Engineer August 2020 - July 2021 Accenture ● Delivered critical features and stability for AstraZeneca’s VeevaCRM solutions as the key developer for iPatient, managing feature implementations and bug fixes under tight deadlines reducing system downtime by 20%. ACADEMIC PROJECTS Stream Processing with Kafka and Samza - Developed and analyzed a real-time data processing system by using Apache Kafka as the messaging system to handle large streams of incoming data and Apache Samza for processing these streams with minimal delay on AWS. Containers: Docker and Kubernetes - Containerized and orchestrated microservices using Docker and Kubernetes on GCP and Azure, enabling scalable and reliable application deployment and management. Machine Learning on the Cloud - Implemented and deployed an end-to-end machine learning pipeline on GCP, enhancing model accuracy through feature engineering and hyperparameter tuning. EDUCATION School of Computer Science, Carnegie Mellon University December 2024 Master of Software Engineering (Courses: Cloud Computing, Software Architecture, WebApp / TA: Engineering Data Intensive Scalable Systems,QA) Kalinga Institute of Industrial Technology July 2020 Bachelor of Computer Science and Engineering AWARDS AND ACHIEVEMENTS ● Silver Award - Bank of America (Q! 2023) ● Top 4 – Accenture x Salesforce Hackathon (2021)'''

def split_text(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

chunks = split_text(full_text)
print(chunks)

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

print("Encoding chunks...")
embeddings = model.encode(chunks)
print("Encoding done!")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

query = "Tell me about my projects"
query_embedding = model.encode([query])
k = 3
distances, indices = index.search(np.array(query_embedding), k)
context = " ".join([chunks[i] for i in indices[0]])
result = subprocess.run(
    ["/opt/homebrew/bin/ollama", "run", "llama2", "--prompt", f"Answer the following question using this context: {context}\nQuestion: {query}"],
    capture_output=True,
    text=True
)
print(result.stdout)