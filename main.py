from contextlib import asynccontextmanager
from pathlib import Path
import os
import pickle
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. PATHS & CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Artifacts" / "bi_gru_model.h5"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer_emotion.pkl"

MAX_SEQUENCE_LENGTH = 50

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. PYDANTIC SCHEMAS
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        },
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


# ============================================================
# 4. MODEL STORAGE
# ============================================================

dl_model = {
    "BiGRU": None,
    "Tokenizer": None,
}


# ============================================================
# 5. MODEL LOADING
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("========================================")
    print("Starting Sentiment Analysis API")
    print("========================================")

    print(f"Model path: {MODEL_PATH}")
    print(f"Tokenizer path: {TOKENIZER_PATH}")

    try:
        # Check whether required files exist
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {TOKENIZER_PATH}"
            )

        print("Loading BiGRU model...")

        # compile=False because this API only performs inference
        dl_model["BiGRU"] = load_model(
            MODEL_PATH,
            compile=False
        )

        print("BiGRU model loaded successfully.")

        print("Loading tokenizer...")

        with open(TOKENIZER_PATH, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)

        print("Tokenizer loaded successfully.")

        print("========================================")
        print("MODEL AND TOKENIZER READY")
        print("========================================")

    except Exception as e:
        print("========================================")
        print("MODEL LOADING FAILED")
        print(f"Error: {repr(e)}")
        print("========================================")

        # Keep FastAPI running even if model loading fails.
        # /health will report model_loaded=False.
        dl_model["BiGRU"] = None
        dl_model["Tokenizer"] = None

    yield

    print("Shutting down API...")

    dl_model["BiGRU"] = None
    dl_model["Tokenizer"] = None


# ============================================================
# 6. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Emotion classification using a BiGRU neural network",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 7. CORS
# ============================================================

# For production, Render can provide the frontend URL through:
#
# FRONTEND_URL=https://your-frontend.onrender.com
#
# For now, "*" allows your deployed React frontend to communicate
# with the API.

frontend_url = os.getenv("FRONTEND_URL")

if frontend_url:
    allowed_origins = [
        origin.strip()
        for origin in frontend_url.split(",")
        if origin.strip()
    ]
else:
    allowed_origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 8. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running",
        "status": "online",
    }


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    model_loaded = (
        dl_model.get("BiGRU") is not None
        and dl_model.get("Tokenizer") is not None
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=model_loaded,
    )


# ============================================================
# 10. PREDICTION ENDPOINT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    # Get model and tokenizer
    bigru_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    # Check whether model is ready
    if bigru_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    # --------------------------------------------------------
    # Preprocess input
    # --------------------------------------------------------

    cleaned_text = preprocess_text(text_input.text)

    # Prevent empty input after preprocessing
    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Text contains no valid characters.",
        )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = bigru_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Find highest probability emotion
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    top_emotion = EMOTION_LABELS[
        top_emotion_index
    ]

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    all_probabilities = {
        label: float(probability)
        for label, probability in zip(
            EMOTION_LABELS,
            probabilities
        )
    }

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS.get(
            top_emotion,
            ""
        ),
        confidence=float(
            probabilities[top_emotion_index]
        ),
        all_probabilities=all_probabilities,
    )