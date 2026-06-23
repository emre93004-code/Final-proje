# -*- coding: utf-8 -*-
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

# ==============================================================================
# --- 1. AYARLAR VE DİZİN OLUŞTURMA ---
# ==============================================================================
output_folder = "rapor_çıktıları"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# NLTK İndirmeleri (Sessiz modda)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# ==============================================================================
# --- 2. VERİ YÜKLEME VE ÖN İŞLEME (PREPROCESSING) ---
# ==============================================================================
def preprocess(text, method='lemma'):
    if not isinstance(text, str): 
        return []
    # Sadece alfabetik karakterleri tut ve küçük harfe çevir
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

# Ana Veri Setini Yükleme
try:
    df = pd.read_csv('all_job_post.csv')
    print("[OK] 'all_job_post.csv' başarıyla yüklendi.")
except FileNotFoundError:
    # Dosya yoksa koruma amaçlı örnek dummy veri seti oluşturulur
    print("[UYARI] 'all_job_post.csv' bulunamadı! Sentetik test verisi üretiliyor...")
    df = pd.DataFrame({
        'job_description': [
            'data scientist role machine learning python pandas optimization',
            'python developer job backend software engineering django framework',
            'ai engineer deep neural networks computer vision natural language',
            'data analyst sql tableau powerbi data cleaning reporting',
            'software dev full stack agile scrum git engineering development'
        ] * 10
    })

# Raporlamaya uygun eşsiz döküman kimlikleri atama
df['doc_id'] = [f"doc_{i+1}" for i in range(len(df))]


# ==============================================================================
# --- 3. 16 FARKLI MODELİN HİPERPARAMETRİK EĞİTİMİ ---
# ==============================================================================
print("\n16 Model varyasyon kombinasyonu oluşturuluyor...")
params = []
for sg in [0, 1]:              # 0: CBOW, 1: Skip-gram
    for win in [2, 4]:         # Pencere Genişliği (Window Size)
        for dim in [100, 300]: # Vektör Boyutu (Vector Dimension)
            for method in ['lemma', 'stem']: # Ön İşleme Stratejisi
                params.append({'sg': sg, 'win': win, 'dim': dim, 'method': method})

models = {}
for p in params:
    method_col = 'lemma_tokens' if p['method'] == 'lemma' else 'stem_tokens'
    
    # Tokenizasyon işlemini sütun yoksa bir kere yap
    if method_col not in df.columns:
        df[method_col] = df['job_description'].apply(lambda x: preprocess(x, p['method']))
    
    # Model ismi formatlama
    name = f"{'SG' if p['sg'] else 'CBOW'}_w{p['win']}_d{p['dim']}_{p['method']}"
    
    # Word2Vec Model Eğitimi
    models[name] = Word2Vec(
        sentences=df[method_col].tolist(), 
        vector_size=p['dim'], 
        window=p['win'], 
        sg=p['sg'], 
        min_count=1, 
        workers=4, 
        epochs=10
    )
    print(f" -> Model Eğitildi: {name}")


# Döküman vektörlerini hesaplama fonksiyonu
def get_vec(model, tokens):
    valid = [t for t in tokens if t in model.wv]
    return np.mean(model.wv[valid], axis=0) if valid else np.zeros(model.vector_size)


# ==============================================================================
# --- 4. BENZERLİK HESAPLAMA VE GÜVENLİ ÇIKTILARIN HAZIRLAN
