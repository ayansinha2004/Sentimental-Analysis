from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for your Render frontend domain and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://emotion-frontends.onrender.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"  # Temporarily allow all origins to test
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Emotion Classification API is running."}