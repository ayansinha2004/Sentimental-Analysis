import os
import pickle
import re
from pathlib import Path
from contextlib import asynccontextmanager

# ============================================================
# TensorFlow configuration
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IMPORTANT:
# Use TensorFlow Keras consistently
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Limit TensorFlow threads for low-memory hosting
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

model_path = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "Artifacts" / "bi_gru_model.h5")
    )
)

tokenizer_path = Path(
    os.getenv(
        "TOKENIZER_PATH",
        str(BASE_DIR / "Artifacts" / "tokenizer.pkl")
    )
)

max_sequence_length = 50


# ============================================================
# Emotion configuration
# ============================================================

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
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
# Text preprocessing
# ============================================================

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Pydantic schemas
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000
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
    error: str | None = None
    tf_version: str | None = None
    keras_version: str | None = None


# ============================================================
# Model state
# ============================================================

dl_model = {}

model_error = None


# ============================================================
# Application lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model_error

    print("=" * 60)
    print("STARTING EMOTION CLASSIFICATION API")
    print("=" * 60)

    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)

    print("Base directory:", BASE_DIR)
    print("Model path:", model_path)
    print("Tokenizer path:", tokenizer_path)

    print("Model exists:", model_path.exists())
    print("Tokenizer exists:", tokenizer_path.exists())

    try:

        # ----------------------------------------------------
        # Check model file
        # ----------------------------------------------------

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        # ----------------------------------------------------
        # Check tokenizer file
        # ----------------------------------------------------

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        print("Loading BiGRU model...")

        dl_model["BiGRU"] = load_model(
            str(model_path),
            compile=False
        )

        print("BiGRU model loaded successfully.")

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        print("Loading tokenizer...")

        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)

        print("Tokenizer loaded successfully.")

        model_error = None

        print("=" * 60)
        print("MODEL + TOKENIZER LOADED SUCCESSFULLY")
        print("API IS READY")
        print("=" * 60)

    except Exception as e:

        model_error = repr(e)

        print("=" * 60)
        print("MODEL LOADING FAILED")
        print("=" * 60)

        print("ERROR:")
        print(repr(e))

        print("=" * 60)

        dl_model.clear()

    yield

    # Cleanup
    dl_model.clear()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Emotion Classification API",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Emotion Classification API Running",
        "health": "/health",
        "predict": "/predict"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    is_loaded = (
        "BiGRU" in dl_model
        and
        "Tokenizer" in dl_model
    )

    return HealthResponse(
        status=(
            "Server is running"
            if is_loaded
            else "Model loading failed"
        ),
        model_loaded=is_loaded,
        error=None if is_loaded else model_error,
        tf_version=tf.__version__,
        keras_version=tf.keras.__version__,
    )


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if BiGRU_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Check the /health endpoint."
            )
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    cleaned_text = preprocess_text(
        text_input.text
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
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = BiGRU_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Top emotion
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    top_emotion = emotion_labels[
        top_emotion_index
    ]

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(
            emotion_labels,
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

        all_probabilities=all_probabilities
    )