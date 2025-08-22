# Minimal working FastAPI app for testing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Stock Prediction API - Test Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Stock Prediction API - Test Version", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "stock-prediction-api"}

@app.get("/version")
async def version():
    return {"version": "1.0.0", "environment": "test"}

@app.get("/test")
async def test():
    return {"message": "API is working!", "port": os.getenv("PORT", "unknown")}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
