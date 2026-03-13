from fastapi import FastAPI 
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from Inference import ask

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):

    return await run_in_threadpool(ask,query.question)

