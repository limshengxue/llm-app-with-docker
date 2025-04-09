import os
from fastapi import FastAPI, Response
from langchain_ollama import ChatOllama
from pydantic import BaseModel

app = FastAPI()

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
model_name = os.environ.get('MODEL_NAME', 'gemma3:1b')

class QueryInput(BaseModel):
    query: str

@app.get('/')
def home():
    return {"Chat" : "Bot"}

@app.post('/query')
def query(input : QueryInput):
    llm = ChatOllama(
        base_url=ollama_url,
        model=model_name,
    )

    messages = [
        (
            "system",
            "Think carefully and answer the question as concisely as possible.",
        ),
        ("human", input.query),
    ]

    ai_msg = llm.invoke(messages)
    
    return {"response" : ai_msg.content}
