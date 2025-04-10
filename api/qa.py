from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from app import vectorstore
from langchain import hub
from langgraph.graph import START, StateGraph
from langchain_ollama import ChatOllama
import os

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
chroma_url = os.environ.get('CHROMA_URL', 'http://vectordb:8000')
model_name = os.environ.get('MODEL_NAME', 'gemma3:1b')

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    print(vectorstore.get()['documents'])
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    llm = ChatOllama(
        base_url=ollama_url,
        model=model_name,
    )

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()