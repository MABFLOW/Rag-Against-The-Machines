from fastapi import FastAPI
from pydantic import BaseModel
from .engine import Engine


class QueryRequest(BaseModel):
    query: str
    k: int


app = FastAPI()
engine = Engine()

@app.post("/query")
def query(request: QueryRequest):
    return engine.search(request.query, request.k)


@app.post("/answer")
def answer(request: QueryRequest):
    return engine.answer(request.query, request.k)