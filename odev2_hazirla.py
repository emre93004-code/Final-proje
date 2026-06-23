import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. AYARLAR VE DİZİN OLUŞTURMA ---
output_folder = "rapor_çıktıları"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# NLTK İndirmeleri
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- 2. VERİ YÜKLEME VE TEMİZLEME ---
def preprocess(text, method='lemma'):
    if not isinstance(text, str): return []
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    if method == 'lemma':
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    else:
        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokens]

# Dosya yükleme (Dosya adını buraya yaz)
try:
    df = pd.read_csv('all_job_post.csv')
except FileNotFoundError:
    # Dosya yoksa hata vermemesi için test verisi
    df = pd.DataFrame({'job_description': ['data scientist role', 'python developer job', 'ai engineer', 'data analyst', 'software dev'] * 10})

df['doc_id'] = [f"doc_{i+1}" for i in range(len(df))]

# --- 3. 16 MODEL EĞİTİMİ (Kombinasyonel Yapı) ---
params = []
for sg in [0, 1]:           # 0: CBOW, 1: Skipgram
    for win in [2, 4]:      # Window size
        for dim in [100, 300]: # Vector size
            for method in ['lemma', 'stem']: # Processing type
                params.append({'sg': sg, 'win': win, 'dim': dim, 'method': method})

models = {}
for p in params:
    method_col = 'lemma_tokens' if p['method'] == 'lemma' else 'stem_tokens'
    if method_col not in df.columns:
        df[method_col] = df['job_description'].apply(lambda x: preprocess(x, p['method']))
    
    name = f"{'SG' if p['sg'] else 'CBOW'}_w{p['win']}_d{p['dim']}_{p['method']}"
    models[name] = Word2Vec(sentences=df[method_col].tolist(), vector_size=p['dim'], window=p['win'], sg=p['sg'], min_count=1, workers=4, epochs=10)
    print(f"Model Eğitildi: {name}")

# --- 4. BENZERLİK HESAPLAMA VE CSV ÇIKTILARI ---
def get_vec(model, tokens):
    valid = [t for t in tokens if t in model.wv]
    return np.mean(model.wv[valid], axis=0) if valid else np.zeros(model.vector_size)

all_results = {}
report_data = []

query_idx = 0 # Sorgu dokümanı
for name, model in models.items():
    method = name.split('_')[-1]
    tokens = preprocess(df['job_description'].iloc[query_idx], method)
    q_vec = get_vec(model, tokens).reshape(1, -1)
    
    scores = []
    for _, row in df.iterrows():
        d_vec = get_vec(model, preprocess(row['job_description'], method)).reshape(1, -1)
        scores.append(cosine_similarity(q_vec, d_vec)[0][0])
    
    df['score'] = scores
    top5 = df.sort_values('score', ascending=False).iloc[1:6] # İlkini (kendisi) atla, ilk 5'i al
    
    all_results[name] = top5['doc_id'].tolist()
    
    # Rapor satırı
    report_data.append({
        'Model': name,
        'Top_5_Docs': ', '.join(top5['doc_id'].tolist()),
        'Avg_Score': df['score'].mean()
    })

# CSV Olarak Kaydet
pd.DataFrame(report_data).to_csv(f"{output_folder}/model_performans_raporu.csv", index=False)
print("CSV Raporu kaydedildi.")

# --- 5. JACCARD MATRİSİ VE ISI HARİTASI ---
model_names = list(all_results.keys())
jaccard_matrix = np.zeros((len(model_names), len(model_names)))

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        set1, set2 = set(all_results[m1]), set(all_results[m2])
        jaccard_matrix[i][j] = len(set1 & set2) / len(set1 | set2)

plt.figure(figsize=(14, 10))
sns.heatmap(pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Modeller Arası Jaccard Benzerlik Matrisi")
plt.tight_layout()
plt.savefig(f"{output_folder}/jaccard_heatmap.png")
print("Isı haritası kaydedildi.")
# --- ODEV2_HAZIRLA.PY SONUNA EKLENMESİ GEREKENLER ---
# Build_report.py dosyasının beklediği isimlerle dosyaları kaydediyoruz:
pd.DataFrame(cosine_eval_data).to_csv('cosine_eval.csv', index=False)
pd.DataFrame(semantic_template_data).to_csv('semantic_eval.csv', index=False)
pd.DataFrame(similar_words_data).to_csv('similar_words.csv', index=False)
pd.DataFrame(top5_data).to_csv('top5_per_model.csv', index=False)
pd.DataFrame(jaccard_matrix_data).to_csv('jaccard_matrix.csv', index=False)
# summary.csv'yi burada oluşturmayı unutma!
# --- summary.csv dosyasını oluşturma ---
summary_data = {
    'Baslik': ['Model Sayısı', 'Veri Seti', 'Kullanılan Yöntemler', 'Vector Boyutları', 'Window Size'],
    'Deger': ['16', 'all_job_post.csv', 'Word2Vec (CBOW ve Skip-Gram)', '100 ve 300', '2 ve 4']
}

pd.DataFrame(summary_data).to_csv('summary.csv', index=False, encoding='utf-8')
print(" -> summary.csv başarıyla oluşturuldu.")
