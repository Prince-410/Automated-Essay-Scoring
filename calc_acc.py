import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

SCORE_RANGE = {1:(2,12), 2:(1,6), 3:(0,3), 4:(0,3),
               5:(0,4), 6:(0,4), 7:(2,24), 8:(10,60)}

df = pd.read_csv('training_set.csv', encoding='utf-8-sig')
df = df.dropna(subset=['domain1_score']).reset_index(drop=True)

embeddings = np.load('essay_embeddings_v2.npy')

def extract_features(text, essay_set):
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    sentence_count = max(len(sentences), 1)
    return [word_count, char_count, sentence_count, 
            char_count/(word_count+1), word_count/sentence_count, 
            len(set(words))/(word_count+1), text.count(','), 
            text.count('.')+text.count('!')+text.count('?'), essay_set]

hand_features = np.array([extract_features(r['essay'], r['essay_set']) for _, r in df.iterrows()])
X = np.hstack([embeddings, hand_features])

model = joblib.load('lgbm_essay_model_refined.pkl')

raw_preds = model.predict(X)

def denorm(p, s_id):
    lo, hi = SCORE_RANGE[s_id]
    return int(np.clip(np.round(p * (hi - lo) + lo), lo, hi))

pred_scores = [denorm(p, s) for p, s in zip(raw_preds, df['essay_set'])]
true_scores = df['domain1_score'].astype(int).values

acc = accuracy_score(true_scores, pred_scores)
print(f"Overall Exact-Match Accuracy: {acc*100:.2f}%")

df['pred'] = pred_scores
df['true'] = true_scores
print("\nAccuracy per Essay Set:")
for s in sorted(SCORE_RANGE.keys()):
    subset = df[df['essay_set'] == s]
    s_acc = (subset['true'] == subset['pred']).mean()
    print(f"Set {s}: {s_acc*100:.2f}%")
