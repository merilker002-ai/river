# model_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import base64
from river import anomaly, preprocessing
from typing import List, Dict
import asyncio
import aiofiles
import os
from contextlib import asynccontextmanager

# Model state
model = None
learning_stats = {
    "total_processed": 0,
    "last_learning": None,
    "model_size": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model from GitHub
    global model
    model = await load_model_from_github()
    yield
    # Shutdown: Save model to GitHub
    await save_model_to_github(model)

app = FastAPI(title="Su Tüketim AI API", lifespan=lifespan)

class LearningRequest(BaseModel):
    data: List[Dict]
    batch_id: str

class PredictionRequest(BaseModel):
    data: Dict

async def load_model_from_github():
    """GitHub'dan modeli async olarak yükle"""
    try:
        # GitHub REST API ile modeli çek
        model_url = f"https://raw.githubusercontent.com/merilker002-ai/river/main/models/river_model.pkl"
        # Bu kısım GitHub API token ile geliştirilebilir
        return preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
            n_estimators=30, height=10, window_size=100, seed=42
        )
    except:
        # Yeni model oluştur
        return preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(
            n_estimators=30, height=10, window_size=100, seed=42
        )

async def save_model_to_github(model_obj):
    """Modeli GitHub'a async olarak kaydet"""
    try:
        # Modeli pickle ile serialize et
        model_bytes = pickle.dumps(model_obj)
        
        # Burada GitHub API ile commit yapılacak
        # Basit versiyon: local'e kaydet (GitHub Actions ile otomatik push)
        async with aiofiles.open('river_model.pkl', 'wb') as f:
            await f.write(model_bytes)
        
        return True
    except Exception as e:
        print(f"Model save error: {e}")
        return False

@app.post("/incremental-learn")
async def incremental_learn(request: LearningRequest):
    """Incremental learning endpoint - BELLEK DOSTU"""
    global model, learning_stats
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        processed_count = 0
        scores = []
        
        # Batch processing - memory efficient
        for record in request.data:
            features = {
                "tuketim": float(record.get('AKTIF_m3', 0)),
                "gunluk_ort": float(record.get('GUNLUK_ORT_TUKETIM_m3', 0)),
                "tutar": float(record.get('TOPLAM_TUTAR', 0))
            }
            
            # INCREMENTAL LEARNING - tek kayıt
            score = model.score_one(features)
            model.learn_one(features)  # ✅ Gerçek incremental
            
            scores.append(score)
            processed_count += 1
            
            # Memory management: Her 1000 kayıtta bir istatistik güncelle
            if processed_count % 1000 == 0:
                await asyncio.sleep(0.01)  # Async break
        
        # Statistics update
        learning_stats["total_processed"] += processed_count
        learning_stats["last_learning"] = datetime.now().isoformat()
        learning_stats["model_size"] = len(pickle.dumps(model))
        
        # Async model save (non-blocking)
        asyncio.create_task(save_model_to_github(model))
        
        return {
            "status": "success",
            "processed": processed_count,
            "batch_id": request.batch_id,
            "avg_score": np.mean(scores) if scores else 0,
            "memory_usage": f"{learning_stats['model_size'] / 1024:.2f} KB"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_anomaly(request: PredictionRequest):
    """Anomali tahmini endpoint"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        features = {
            "tuketim": float(request.data.get('AKTIF_m3', 0)),
            "gunluk_ort": float(request.data.get('GUNLUK_ORT_TUKETIM_m3', 0)),
            "tutar": float(request.data.get('TOPLAM_TUTAR', 0))
        }
        
        score = model.score_one(features)
        
        return {
            "score": score,
            "risk_level": "Yüksek" if score > 0.7 else "Orta" if score > 0.4 else "Düşük",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Model bilgilerini getir"""
    return {
        "status": "active" if model else "inactive",
        "stats": learning_stats,
        "model_type": "River HalfSpaceTrees",
        "last_update": learning_stats.get("last_learning", "Never")
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)