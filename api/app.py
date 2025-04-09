import requests
import os
from fastapi import FastAPI, Response

app = FastAPI()

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
model_name = os.environ.get('MODEL_NAME', 'gemma3:1b')

@app.get('/')
def home():
    return {"Chat" : "Bot"}

@app.get('/ask')
def ask(prompt :str):
    res = requests.post(ollama_url + '/api/generate', json={
        "prompt": prompt,
        "stream" : False,
        "model" : model_name
    })

    return Response(content=res.text, media_type="application/json")