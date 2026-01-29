# ==============================
# IMPORTS
# ==============================
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import os


# ==============================
# LLM & EMBEDDINGS 
# ==============================
llm = ChatOllama(model="llama3.1", temperature=0)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ==============================
# STATE DEFINITION
# ==============================
class ResumeState(TypedDict):
    resume_path: str
    jb_path: str

    resume_text: str 
    jb_text: str 

    resume_structured: Dict
    jb_structured: Dict

    similarity_score: Dict
    final_score: float

    gaps: str
    suggestions: str


# ==============================
# UTILS
# ==============================
def load_pdf_text(file) -> str:
    """
    Accepts either a path (str) or a BytesIO uploaded file from Streamlit.
    """
    if hasattr(file, "read"):  # BytesIO or file-like
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
    else:
        loader = PyPDFLoader(file)

    pages = loader.load()
    return "\n".join([p.page_content for p in pages])



# ==============================
# NODE 1: LOAD DOCUMENTS
# ==============================
def load_documents(state: ResumeState):
    resume_text = load_pdf_text(state["resume_path"])
    jb_text = load_pdf_text(state["jb_path"])

    return {
        "resume_text": resume_text,
        "jb_text": jb_text
    }


# ==============================
# NODE 2: STRUCTURE DOCUMENTS
# ==============================
structure_prompt = ChatPromptTemplate.from_template("""
You are an expert resume analyzer.

Extract structured information from the following text.

TEXT:
{input_text}

Return STRICT JSON with:
- skills (list)
- experience (list of short points)
- projects (list)
- education (list)
""")
def structure_document(state: ResumeState):
    resume_structured = llm.invoke(
        structure_prompt.format(input_text=state["resume_text"])
    ).content

    jb_structured = llm.invoke(
        structure_prompt.format(input_text=state["jb_text"])
    ).content

    return {
        "resume_structured": resume_structured,
        "jb_structured": jb_structured
    }


# ==============================
# NODE 3: SIMILARITY SCORING
# ==============================
def similarity_score_node(state: ResumeState):
    
    sections = ["skills", "experience", "projects"]
    scores = {}

    for section in sections:
        resume_vec = embeddings.embed_query(
            state["resume_structured"]
        )

        jb_vec = embeddings.embed_query(
            state["jb_structured"]
        )

        score = cosine_similarity(
            [resume_vec],
            [jb_vec]
        )[0][0]

        scores[section] = round(score, 3)

    return {
        "similarity_score": scores
    }


# ==============================
# NODE 4: FINAL SCORE
# ==============================
def final_score_node(state: ResumeState):

    weights = {
        "skills": 0.4,
        "experience": 0.35,
        "projects": 0.25
    }

    final_score = sum(
        state["similarity_score"][k] * v
        for k, v in weights.items()
    )

    return {"final_score": round(final_score * 100, 2)}


# ==============================
# NODE 5: GAP ANALYSIS
# ==============================
def gap_analysis_node(state: ResumeState):
    prompt = f"""
Compare the resume and job description.

Resume (Structured):
{state["resume_structured"]}

Job Description (Structured):
{state["jb_structured"]}

Identify:
- Missing skills
- Weak experience areas
- ATS keyword gaps
"""
    gaps = llm.invoke(prompt).content

    return {"gaps": gaps}


# ==============================
# NODE 6: IMPROVEMENT SUGGESTIONS
# ==============================
def suggestion_node(state: ResumeState):
    prompt = f"""
Resume Match Score: {state["final_score"]}

Gaps Identified:
{state["gaps"]}

Suggest:
1. Resume improvements
2. Better bullet points
3. Skills to add
4. ATS optimization tips
"""
    suggestions = llm.invoke(prompt).content

    return {"suggestions": suggestions}

graph = StateGraph(ResumeState)

graph.add_node("load", load_documents)
graph.add_node("structure", structure_document)
graph.add_node("similarity_score", similarity_score_node)
graph.add_node("final_score", final_score_node)
graph.add_node("gaps", gap_analysis_node)
graph.add_node("suggestion", suggestion_node)

graph.add_edge(START, "load")
graph.add_edge("load", "structure")
graph.add_edge("structure", "similarity_score")
graph.add_edge("similarity_score", "final_score")
graph.add_edge("final_score", "gaps")
graph.add_edge("gaps", "suggestion")
graph.add_edge("suggestion", END)


app = graph.compile()
