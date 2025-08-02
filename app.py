from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from main import run_retrieval

app = FastAPI()

class Query(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def post_query(query: Query, request: Request):
    # auth = request.headers.get("Authorization")
    # if not auth or not auth.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        result = run_retrieval(query.documents, query.questions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))