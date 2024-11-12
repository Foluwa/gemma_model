# ./app/main.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
# from tensorflow.keras.models import load_model
import json

# Set up environment
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define FastAPI app
app = FastAPI()

# Define GemmaQA prompt template
template = "\n\n<|system|>:\n{instruct}\n\n<|user|>:\n{question}\n\n<|assistant|>:\n{answer}"

# Load the model and define GemmaQA class
class GemmaQA:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.prompt = template
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
    def load_model(self):
        # Load the model directly from the saved .h5 file or saved model directory
        # model_path = "/app/app/gemma2_2b_en_lpi/model.weights.h5"
        # model = load_model(model_path)
        # model_path = "/app/app/gemma2_2b_en_lpi"
        model_path = "/app/gemma2_2b_en_lpi"
        model = tf.keras.models.load_model(model_path)
        return model

    def load_tokenizer(self):
        # Load tokenizer configuration from tokenizer.json
        tokenizer_path = "/app/gemma2_2b_en_lpi/tokenizer.json"
        with open(tokenizer_path, 'r') as f:
            tokenizer_config = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
        return tokenizer

    def preprocess_input(self, input_text):
        # Tokenize and pad the input text
        tokens = self.tokenizer.texts_to_sequences([input_text])
        tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=self.max_length, padding='post')
        return tokens

    def query(self, instruct, question):
        input_text = self.prompt.format(instruct=instruct, question=question, answer="")
        tokens = self.preprocess_input(input_text)
        response = self.model.predict(tokens)
        # Convert the response to text (this may need to be adjusted based on the model output)
        return self.tokenizer.sequences_to_texts(response)[0].split("<|assistant|>:")[-1].strip()

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
    with open("/app/app/templates/frontend.html") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
