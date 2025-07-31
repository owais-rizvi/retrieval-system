import requests
from sentence_transformers import SentenceTransformer
import faiss
import pymupdf
import google.generativeai as genai # New import for Gemini API
import os
# from google.colab import userdata # NEW: Import userdata
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. "
                     "Please ensure it is defined in your .env file")
# --- Configure Gemini API ---
try:
    # Use google.colab.userdata to directly retrieve the secret
    # GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")

    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        # This case should ideally be caught by userdata.get() raising an error,
        # but as a fallback
        raise ValueError("Google API Key found in secrets but is empty. Please check its value.")

except Exception as e:
    # Catch any error from userdata.get(), e.g., SecretNotFoundError
    raise ValueError(f"Failed to retrieve Google API Key from Colab Secrets: {e}. "
                     "Please ensure the secret 'GOOGLE_API_KEY' exists and "
                     "Notebook access is enabled in the 'Secrets' panel.")

# The rest of your code remains the same...

# Step 1: Download the PDF from a URL
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

# Step 2: Parse the PDF content using PyMuPDF
def parse_pdf(pdf_content):
    doc = pymupdf.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Step 3: Generate sentence embeddings for the queries and document
def generate_embeddings(texts):
    if not hasattr(generate_embeddings, "model"):
        generate_embeddings.model = SentenceTransformer('all-MiniLM-L6-v2')
    return generate_embeddings.model.encode(texts)

# Step 4: Split document into logical clauses or sections
def split_document_into_clauses(document_text):
    clauses = []
    # Attempt to split by lines, then by sentences within lines if too long
    lines = document_text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        # If a line is too long, try splitting into sentences
        if len(stripped_line) > 200 and '. ' in stripped_line:
            sentences = [s.strip() for s in stripped_line.split('. ') if s.strip()]
            clauses.extend(sentences)
        else:
            clauses.append(stripped_line)
    # Further filter out very short or non-informative clauses
    return [clause for clause in clauses if len(clause) > 20]


# Step 5: Create FAISS index for the document clauses
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Step 6: Perform semantic search to match the query with the document clauses
def match_clauses(query, document_text, k=10): # Increased k to get more context
    clauses = split_document_into_clauses(document_text)
    if not clauses:
        return []

    # Filter out clauses that are too short to embed meaningfully or contain only numbers/symbols
    filtered_clauses = [c for c in clauses if len(c) > 10 and any(char.isalpha() for char in c)]

    if not filtered_clauses:
        return []

    clause_embeddings = generate_embeddings(filtered_clauses)
    query_embedding = generate_embeddings([query])

    faiss_index = create_faiss_index(clause_embeddings)

    # Ensure query_embedding is 2D, even for a single query
    D, I = faiss_index.search(query_embedding, k=min(k, len(filtered_clauses)))
    matched_clauses = [filtered_clauses[i] for i in I[0]]
    return matched_clauses

# Step 7: Evaluate the matched clauses to extract the answer (using Gemini API)
def evaluate_logic(matched_clauses, question):
    # Initialize the Gemini model
    # Use 'gemini-1.5-flash-latest' for a good balance of speed and capability with a large context window
    # Or 'gemini-1.5-pro-latest' for higher quality but potentially slower/more expensive
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Combine matched clauses into a single context for the LLM
    # Gemini models have very large context windows, so we can concatenate more clauses.
    # It's good practice to mark the start/end of the context clearly for the LLM.
    context = "\n\n".join(matched_clauses)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.

Document Context:
---
{context}
---

Question: {question}

Instructions:
- Provide a concise and direct answer to the question.
- If the answer is not explicitly stated in the provided Document Context, respond with "The document does not contain a direct answer to this question."
- Do not make up information.
- Be as specific as possible.
"""
    try:
        # Use generate_content for chat-like interactions
        response = model.generate_content(prompt)
        # Access the text content of the response
        answer = response.text.strip()

        # We can't get a "score" directly from generative models like Gemini API in the same way
        # an extractive QA model provides it. We can infer a "confidence" based on whether
        # it says it can't find the answer.
        score = 1.0 if "document does not contain a direct answer" not in answer.lower() else 0.0

        return [{"answer": answer, "score": score}]

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return [{"answer": f"An error occurred while trying to find the answer: {e}", "score": 0.0}]

# Step 8: Format the response in the required JSON format
def format_json_response(answers):
    response = {
        "answers": answers
    }
    return response


    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

    try:
        print("Downloading PDF...")
        pdf_content = download_pdf(url)
        print("Parsing PDF content...")
        document_text = parse_pdf(pdf_content)
        print("Document text extracted successfully.")

        questions = [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]

        all_responses = []
        for i, question in enumerate(questions):
            print(f"\nProcessing Question {i+1}: \"{question}\"")
            matched_clauses = match_clauses(question, document_text, k=15) # Increased k further for Gemini
            if not matched_clauses:
                print("No relevant clauses found for this question.")
                answers = [{"answer": "No relevant information found in the document.", "score": 0.0}]
            else:
                # print(f"Top {len(matched_clauses)} matched clauses for context:")
                # for j, clause in enumerate(matched_clauses):
                #     print(f"  [{j+1}] {clause[:100]}...") # Print first 100 chars
                answers = evaluate_logic(matched_clauses, question)

            response = format_json_response(answers)
            all_responses.append({
                "question": question,
                "response": response
            })
            print(f"Answer: {answers[0]['answer']} (Score: {answers[0]['score']:.2f})")

        import json
        print("\n--- All Responses ---")
        print(json.dumps(all_responses, indent=2))

    except Exception as e:
        print(f"Error: {e}")