import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
import chromadb
from uuid import uuid4
from langchain_core.documents import Document
from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_collection()
    yield

load_dotenv()
app = FastAPI(lifespan=lifespan)

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
chroma_url = os.environ.get('CHROMA_URL', 'http://vectordb:8000')
model_name = os.environ.get('MODEL_NAME', 'gemma3:1b')

embeddings = MistralAIEmbeddings(
            model="mistral-embed",
        )

chroma_client = chromadb.HttpClient(host='vectordb', port=8000)
vectorstore = Chroma(
    client=chroma_client,
    collection_name="company_policies",
    embedding_function=embeddings,
)

class QueryInput(BaseModel):
    query: str

@app.get('/')
def home():
    return "FastAPI server now running"

def init_collection():
    if len(vectorstore.get()['documents']) == 0:
        documents = [
            Document(
                page_content="The working hours for the employee is 6 hours per day",
                metadata={"source": "tweet"},
                id=1,
            ),
            Document(
                page_content="The annual leave for an employee is 16 days",
                metadata={"source": "tweet"},
                id=2,
            ),
        ]
        uuids = [str(uuid4()) for _ in range(len(documents))]

        vectorstore.add_documents(documents=documents, ids=uuids)



@app.post('/query')
def query(input : QueryInput):
    from qa import graph
    result = graph.invoke({"question": input.query})

    return {"response" : result["answer"], "context" : result["context"]}
