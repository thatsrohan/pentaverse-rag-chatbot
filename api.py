from fastapi import FastAPI 
from pydantic import BaseModel

from Inference import ask

app = FastAPI()

class Query(BaseModel):
    question: str

app.post("/chat")
def chat(query: Query):

    return ask(query.question)

