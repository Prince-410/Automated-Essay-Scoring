from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="AI Essay Scorer Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCORE_RANGE = {1:(2,12), 2:(1,6), 3:(0,3), 4:(0,3),
               5:(0,4), 6:(0,4), 7:(2,24), 8:(10,60)}

print("Loading AI Models for Pro Analytics...")
lgbm_model = joblib.load('lgbm_essay_model_refined.pkl')
st_model = SentenceTransformer("all-mpnet-base-v2")
st_model.max_seq_length = 256
print("Pro AI Server Ready!")

class EssayRequest(BaseModel):
    text: str
    essay_set: int

def extract_detailed_metrics(text, essay_set):
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    char_count = len(text)
    
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    avg_word_len = char_count / (word_count + 1)
    avg_sent_len = word_count / sentence_count
    lex_div = unique_words / (word_count + 1)
    
    commas = text.count(',')
    puncts = len(re.findall(r'[.!?]', text))
    
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "lexical_diversity": round(lex_div * 100, 1), # as percentage
        "punctuation_count": puncts,
        "comma_usage": commas,
        "essay_set": essay_set
    }

@app.get("/")
async def get_index():
    return FileResponse('index.html')

@app.get("/ai_scoring_background_1776032733963.png")
async def get_bg():
    return FileResponse('ai_scoring_background_1776032733963.png')

@app.post("/predict")
async def predict(request: EssayRequest):
    try:
        metrics = extract_detailed_metrics(request.text, request.essay_set)

        emb = st_model.encode([request.text], normalize_embeddings=True)
        hand = np.array([[
            metrics["word_count"], 
            len(request.text), 
            metrics["sentence_count"], 
            metrics["avg_word_length"], 
            metrics["avg_sentence_length"], 
            metrics["lexical_diversity"]/100, 
            metrics["comma_usage"], 
            metrics["punctuation_count"], 
            request.essay_set
        ]])
        
        X = np.hstack([emb, hand])
        norm_score = lgbm_model.predict(X)[0]
        
        lo, hi = SCORE_RANGE[request.essay_set]
        final_score = int(np.clip(np.round(norm_score * (hi - lo) + lo), lo, hi))
        
        return {
            "score": final_score,
            "min_score": lo,
            "max_score": hi,
            "metrics": metrics,
            "normalized_score": float(norm_score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
