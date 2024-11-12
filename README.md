# Gemma Chatbot

This is a chatbot built using a fine-tuned Gemma model from Keras_NLP, served via FastAPI. The chatbot can respond to user queries based on provided instructions and questions.

## Project Structure

- `main.py`: Main FastAPI application file with endpoints for chat and serving the UI.
- `templates/index.html`: HTML and JavaScript for the frontend interface of the chatbot.
- `gemma2_2b_en_lpi-keras-gemma2_2b_en_lpi-v1/`: Directory containing the fine-tuned Gemma model and associated files.
  - `assets/tokenizer/vocabulary.spm`: Vocabulary file for the tokenizer.
  - `config.json`: Configuration file for the model.
  - `metadata.json`: Metadata for the model.
  - `model.weights.h5`: Weights file for the model.
  - `preprocessor.json`: Preprocessor configuration.
  - `task.json`: Task-specific settings.
  - `tokenizer.json`: Tokenizer configuration.

## Prerequisites

- Python 3.8 or later
- TensorFlow 2.x
- FastAPI
- Uvicorn
- keras_nlp

## Setup and Running the Chatbot

1. Install the dependencies:
   ```bash
   pip install fastapi uvicorn


gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001 app.main:app
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app

venv_gemma/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001 app.main:app


export TF_ENABLE_ONEDNN_OPTS=0
