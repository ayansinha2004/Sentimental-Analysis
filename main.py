import os
import pickle
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""
1. Constants
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
model_path = os.getenv("MODEL_PATH", "Artifacts/bi_gru_model.h5")
tokenizer_path = os.getenv("TOKENIZER_PATH", "Artifacts/tokenizer (1).pkl")
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

"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
"""

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


"""
3. Request and Response Schemas
"""

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


"""
4. Model Loading and Lifespan Management
Load the model and tokenizer once the server starts up.
"""
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    try:
        dl_model["BiGRU"] = load_model(model_path)
        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)
        print("Model and tokenizer loaded successfully...")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

    yield

    dl_model.clear()


"""
5. App setup
"""
app = FastAPI(
    title="Emotion Classification API",
    lifespan=lifespan
)

# CORS setup accepting origins dynamically from environment variable
frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = [frontend_url] if frontend_url != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
6. API Endpoints
"""

@app.get("/")
def root():
    return {
        "message": "Emotion Classification API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }


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
            detail="Model is not loaded yet. Please try again later."
        )

    # 1. Clean text
    cleaned_text = preprocess_text(text_input.text)

    # 2. Tokenize + pad
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # 3. Predict
    probabilities = BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index = int(np.argmax(probabilities))
    top_emotion = emotion_labels[top_emotion_index]

    all_probabilities = {
        label: float(prob) for label, prob in zip(emotion_labels, probabilities)
    }

    # 4. Return response
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )