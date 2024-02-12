from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from transformers import pipeline


class GPTNeoModel:
    def __init__(self):
        self.pipe = None
    def load(self):
        self.pipe = pipeline("text-generation", model="EleutherAI/gpt-neo-125m",device=0)

    def generate(self, prompt: str, max_length: int = 50):
        return self.pipe(prompt, max_length=max_length)[0]['generated_text']
    
class InferRequest(BaseModel):
    prompt: str
    max_length: int = 50

app = FastAPI()
model = GPTNeoModel()
model.load()

@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

@app.post("/generate")
def generate(request: InferRequest):
    try:
        prompt = request.prompt
        max_length = request.max_length
        return model.generate(prompt, max_length)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
