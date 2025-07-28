from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    documents: str
    questions: list[str]
    
base_url = "/api/v1"
@app.post(base_url + "/hackrx/run")
def post_query(query: Query):
    try:
        questions = query.questions
        return {
        "answer": f"This is an answer to '{questions[1]}'",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))