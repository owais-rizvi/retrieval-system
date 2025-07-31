from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from main import download_pdf, parse_pdf, match_clauses, evaluate_logic

app = FastAPI()

class Query(BaseModel):
    documents: str
    questions: List[str]

base_url = "/api/v1"

@app.post(base_url + "/hackrx/run")
def post_query(query: Query):
    try:
        # Step 1: Download and parse the PDF
        pdf_content = download_pdf(query.documents)
        document_text = parse_pdf(pdf_content)

        # Step 2: Answer each question and collect the best answer
        flat_answers = []
        for question in query.questions:
            matched_clauses = match_clauses(question, document_text, k=15)

            if not matched_clauses:
                flat_answers.append("No relevant information found in the document.")
            else:
                evaluated = evaluate_logic(matched_clauses, question)
                best_answer = evaluated[0]["answer"]
                flat_answers.append(best_answer)

        # Step 3: Return the list of answers as per required format
        return {"answers": flat_answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
