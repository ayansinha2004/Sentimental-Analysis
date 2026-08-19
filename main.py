import os
import pickle
import re
from contextlib import asynccontextmanager

# 1. Optimize TensorFlow memory & threading for free-tier hosting (512MB RAM limit)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 2. Path Configurations
model_path = os.getenv("MODEL_PATH", "Artifacts/bi_gru_model.h5")
tokenizer_path = os.getenv("TOKENIZER_PATH", "Artifacts/tokenizer.pkl")
max_sequence_length = 50

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

# 3. Preprocessing Helper
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 4. Schemas
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# 5. Model Lifecycle State
dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")
    try:
        dl_model["BiGRU"] = load_model(model_path)
        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
    yield
    dl_model.clear()

app = FastAPI(title="Emotion Classification API", lifespan=lifespan)

# 6. Unrestricted CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7. Endpoints
@app.get("/")
def root():
    return {"message": "API Running", "health": "/health", "predict": "/predict"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    is_loaded = "BiGRU" in dl_model and "Tokenizer" in dl_model
    return HealthResponse(
        status="Server is running" if is_loaded else "Model loading or unavailable",
        model_loaded=is_loaded
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please wait a moment."
        )

    cleaned_text = preprocess_text(text_input.text)
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text, maxlen=max_sequence_length, padding="post", truncating="post"
    )

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    top_emotion = emotion_labels[top_emotion_index]

    all_probabilities = {
        label: float(prob) for label, prob in zip(emotion_labels, probabilities)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )