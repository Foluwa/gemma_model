# ./app/main.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import keras_nlp

# Set up environment
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define FastAPI app
app = FastAPI()

# Define GemmaQA prompt template
template = "\n\n<|system|>:\n{instruct}\n\n<|user|>:\n{question}\n\n<|assistant|>:\n{answer}"

# Load Gemma model and define GemmaQA class
class GemmaQA:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.prompt = template
        self.gemma_causal_lm = keras_nlp.models.GemmaCausalLM.from_preset("./app/models/gemma2_2b_en_lpi")

    def query(self, instruct, question):
        input_text = self.prompt.format(instruct=instruct, question=question, answer="")
        response = self.gemma_causal_lm.generate(input_text, max_length=self.max_length)
        return response.split("<|assistant|>:")[-1].strip()

# Instantiate the GemmaQA model
gemma_qa = GemmaQA(max_length=128)

# Define a Pydantic model for the chat request
class ChatRequest(BaseModel):
    instruction: str
    question: str

# Define the chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    response_text = gemma_qa.query(request.instruction, request.question)
    return JSONResponse(content={"response": response_text})

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("./app/templates/frontend.html") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
