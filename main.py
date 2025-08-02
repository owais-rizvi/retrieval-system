import requests
from sentence_transformers import SentenceTransformer
import faiss
import pymupdf
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please ensure it is defined in your .env file")
genai.configure(api_key=GOOGLE_API_KEY)

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

def parse_pdf(pdf_content):
    doc = pymupdf.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def generate_embeddings(texts):
    if not hasattr(generate_embeddings, "model"):
        generate_embeddings.model = SentenceTransformer('all-MiniLM-L6-v2')
    return generate_embeddings.model.encode(texts)

def split_document_into_clauses(document_text):
    clauses = []
    lines = document_text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if len(stripped_line) > 200 and '. ' in stripped_line:
            sentences = [s.strip() for s in stripped_line.split('. ') if s.strip()]
            clauses.extend(sentences)
        else:
            clauses.append(stripped_line)
    return [clause for clause in clauses if len(clause) > 20]

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def match_clauses(query, document_text, k=25):
    clauses = split_document_into_clauses(document_text)
    if not clauses:
        return []
    filtered_clauses = [c for c in clauses if len(c) > 10 and any(char.isalpha() for char in c)]
    if not filtered_clauses:
        return []
    clause_embeddings = generate_embeddings(filtered_clauses)
    query_embedding = generate_embeddings([query])
    faiss_index = create_faiss_index(clause_embeddings)
    D, I = faiss_index.search(query_embedding, k=min(k, len(filtered_clauses)))
    matched_clauses = [filtered_clauses[i] for i in I[0]]
    return matched_clauses

def optimize_query_with_llm(original_question):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""Given the user's query, rephrase it into a concise and optimized search query that would be most effective for a semantic search engine to find relevant information in a document. Focus on key terms, entities, and the core intent for document retrieval.

    Original Query: "{original_question}"

    Example 1:
    Original Query: "What's the capital of France, and when was it founded?"
    Optimized Search Query: "capital of France, Paris founding date"

    Your optimized search query (concise, direct terms or questions, suitable for semantic search):
    """
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        optimized_query = response.text.strip()
        return optimized_query
    except Exception as e:
        print(f"Error optimizing query with LLM: {e}. Using original query as fallback.")
        return original_question

def evaluate_logic(matched_clauses, question):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    context = "\n\n".join(matched_clauses)
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.

Document Context:
---
{context}
---

Question: {question}

Instructions:
- Provide a detailed, comprehensive, and well-structured answer to the question.
- Do not make up any information that is not present in the Document Context.
- If the answer requires information not explicitly stated in the provided context, state that the document does not contain the required information.
- Synthesize information from multiple parts of the context if necessary to create a complete answer.
- Ensure the answer is easy to read and understand.
"""
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        return answer
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"An error occurred while trying to find the answer: {e}"

def format_json_response(answers_list):
    return {"answers": answers_list}

# Main entry point for FastAPI
def run_retrieval(documents: str, questions: list):
    pdf_content = download_pdf(documents)
    document_text = parse_pdf(pdf_content)
    final_answers = []
    for original_question in questions:
        optimized_query = optimize_query_with_llm(original_question)
        matched_clauses = match_clauses(optimized_query, document_text, k=25)
        if not matched_clauses:
            answer = "No relevant information found in the document."
        else:
            answer = evaluate_logic(matched_clauses, original_question)
        final_answers.append(answer)
    return format_json_response(final_answers)