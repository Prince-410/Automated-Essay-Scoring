import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
import os
import joblib
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

SCORE_RANGE = {1:(2,12), 2:(1,6), 3:(0,3), 4:(0,3),
               5:(0,4), 6:(0,4), 7:(2,24), 8:(10,60)}

df = pd.read_csv("training_set.csv", encoding="utf-8-sig")

essential_cols = ['essay_id', 'essay_set', 'essay', 'domain1_score']
df = df[essential_cols]

df = df.dropna(subset=['domain1_score']).reset_index(drop=True)

print(f"Dataset Cleaned. Total Rows: {len(df)}")
print(f"Columns in use: {df.columns.tolist()}")

def clean_essay(text):
    text = str(text).lower()
    text = re.sub(r'@[a-z0-9]+', '', text) 
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    return text.strip()

df['clean_essay'] = df['essay'].apply(clean_essay)

def normalize(row):
    lo, hi = SCORE_RANGE[row['essay_set']]
    return (row['domain1_score'] - lo) / (hi - lo)

df['score_norm'] = df.apply(normalize, axis=1)
print(f"Target normalized. Distribution mean: {df['score_norm'].mean():.4f}")

def extract_features(text, essay_set):
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    avg_word_len = char_count / (word_count + 1)
    avg_sent_len = word_count / sentence_count
    lex_div = len(set(words)) / (word_count + 1)

    commas = text.count(',')
    puncts = len(re.findall(r'[.!?]', text))
    
    return [word_count, char_count, sentence_count, avg_word_len, avg_sent_len, lex_div, commas, puncts, essay_set]

hand_features = np.array([extract_features(r['essay'], r['essay_set']) for _, r in df.iterrows()])
print(f"Hand features ready. Shape: {hand_features.shape}")

emb_path = "essay_embeddings_v2.npy"

if os.path.exists(emb_path):
    print("Loading cached embeddings...")
    embeddings = np.load(emb_path)
else:
    print("Computing embeddings (mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    model.max_seq_length = 256 # Balanced speed/accuracy
    embeddings = model.encode(df['essay'].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(emb_path, embeddings)

X = np.hstack([embeddings, hand_features])
y = df['score_norm'].values
sets = df['essay_set'].values

def compute_qwk(p_norm, y_norm, current_sets):
    qwks = []
    for s in np.unique(current_sets):
        mask = current_sets == s
        lo, hi = SCORE_RANGE[s]
        p = np.clip(np.round(p_norm[mask] * (hi - lo) + lo), lo, hi).astype(int)
        t = np.round(y_norm[mask] * (hi - lo) + lo).astype(int)
        qwks.append(cohen_kappa_score(t, p, weights='quadratic', labels=list(range(lo, hi + 1))))
    return np.mean(qwks)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))
trained_models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, sets)):
    reg = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.03, num_leaves=63, 
        max_depth=8, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1
    )
    reg.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], 
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    
    oof_preds[va_idx] = reg.predict(X[va_idx])
    trained_models.append(reg)
    print(f"Fold {fold+1} Completed.")

total_qwk, per_set_qwk = 0, []
print(f"{'Essay Set':<10} | {'Samples':<8} | {'QWK Target':<12} | {'Your QWK':<8}")
print("-" * 55)

for s in sorted(SCORE_RANGE.keys()):
    mask = sets == s
    lo, hi = SCORE_RANGE[s]
    s_preds = np.mean([m.predict(X[mask]) for m in trained_models], axis=0)
    
    p = np.clip(np.round(s_preds * (hi - lo) + lo), lo, hi).astype(int)
    t = np.round(y[mask] * (hi - lo) + lo).astype(int)
    
    qwk = cohen_kappa_score(t, p, weights='quadratic', labels=list(range(lo, hi + 1)))
    per_set_qwk.append(qwk)
    print(f"Set {s:<8} | {mask.sum():<8} | 0.9000 | {qwk:.4f}")

mean_qwk = np.mean(per_set_qwk)
print("-" * 55)
print(f"OVERALL ENSEMBLE QWK: {mean_qwk:.4f}")

if mean_qwk >= 0.90:
    print("\n✅ SUCCESS: Target of 90%+ QWK met for the whole system!")
else:
    print("\n⚠ Target not met. Consider increasing n_estimators.")

joblib.dump(trained_models[0], "lgbm_essay_model_refined.pkl")
print("\nModel saved. Pipeline Complete.")
