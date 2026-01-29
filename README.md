🧠 AI Resume Analyzer (LangGraph + LLM)

An AI-powered Resume Analyzer that compares a candidate’s resume with a job description, calculates a match score, identifies gaps, and provides ATS-optimized improvement suggestions using LLMs, LangGraph, and embeddings.

Built with LangGraph, LangChain, Ollama (LLaMA 3.1), HuggingFace embeddings, and Streamlit.

🚀 Features

📄 Upload Resume PDF and Job Description PDF

🧩 Extract structured data (skills, experience, projects, education)

📊 Compute semantic similarity score using embeddings

🎯 Generate final weighted match score

🔍 Identify:

Missing skills

Weak experience areas

ATS keyword gaps

✨ Get AI-generated resume improvement suggestions

🖥️ Simple and clean Streamlit UI

🏗️ Architecture Overview
Resume PDF ─┐
            ├──▶ Load Documents
Job PDF  ───┘
                ↓
        Structure with LLM
                ↓
        Embedding Similarity
                ↓
        Weighted Final Score
                ↓
          Gap Analysis
                ↓
       Improvement Suggestions


Powered by LangGraph state-based workflow.

🧰 Tech Stack

Python

LangGraph

LangChain

Ollama (LLaMA 3.1)

HuggingFace Embeddings

Sentence Transformers

Scikit-learn

Streamlit

📦 Installation
1️⃣ Clone the repository
https://github.com/Haseeblaghari/ai-resume-analyzer.git
cd ai-resume-analyzer

2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🦙 Ollama Setup (Required)

Make sure Ollama is installed and running:

ollama pull llama3.1


Verify:

ollama list

▶️ Run the Application
streamlit run app.py


Then open your browser at:

http://localhost:8501

📂 Project Structure
.
├── Resume_analyzer.py      # LangGraph workflow
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md

🧠 How Scoring Works
Section	Weight
Skills	40%
Experience	35%
Projects	25%

Final Score = Weighted cosine similarity × 100

📌 Example Output

✅ Resume Match Score: 78.4%

❌ Missing Skills: Docker, Kubernetes

⚠️ Weak Areas: Project impact statements

✨ Suggestions:

Add quantified achievements

Improve ATS keywords

Optimize bullet points

🔮 Future Improvements

JSON validation for structured outputs

Section-wise embeddings (skills vs skills)

Resume rewriting feature

Multi-job comparison

Cloud deployment (FastAPI + Docker)

🤝 Contributing

Contributions, issues, and feature requests are welcome.
Feel free to fork and submit a PR.

👤 Author

Haseeb Laghari
AI Engineer | LLM & LangGraph Enthusiast
