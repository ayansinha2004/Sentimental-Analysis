from contextlib import asynccontextmanager
import pickle
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Constants
model_path = "Artifacts/bi_gru_model.h5"
tokenizer_path = "Artifacts/tokenizer_emotion.pkl"
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


# 2. Text Preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Schemas
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
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


# 4. Lifespan Model Loading
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    dl_model["BiGRU"] = load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Model loaded successfully...")

    yield

    dl_model.clear()


# 5. FastAPI App & CORS Setup
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to React origin (e.g. "http://localhost:5173") in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 6. API Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running", model_loaded=bool(dl_model)
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    cleaned_text = preprocess_text(text_input.text)
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )

    probabilities = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilities))
    top_emotion = emotion_labels[top_emotion_index]

    all_probabilities = {
        label: float(prob)
        for prob, label in zip(probabilities, emotion_labels)
    }

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS.get(top_emotion, ""),
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities,
    )