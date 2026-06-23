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
nltk.download('punkt_tab', quiet=True)  # Bu satırı ekledik
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
    print("[UYARI] 'all_job_post.csv' bulunamadı! Sentetik test verisi üretiliyor...")
    df = pd.DataFrame({
        'job_description': [
            'senior business partner talent acquisition experience human resources management',
            'python developer job backend software engineering django framework development',
            'ai engineer deep neural networks computer vision natural language processing',
            'data analyst sql tableau powerbi data cleaning reporting insights',
            'software dev full stack agile scrum git engineering development infrastructure'
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
# --- 4. BENZERLİK HESAPLAMA VE GÜVENLİ ÇIKTILARIN HAZIRLANMASI ---
# ==============================================================================
print("\nBenzerlik skorları hesaplanıyor ve veri havuzu genişletiliyor...")

all_results = {}
final_similarity_rows = []

# --- SORGU TANIMLAMA ---
# Sistem bu İK kriterini alıp veri setindeki tüm ilanlarla tek tek kıyaslayacak
custom_query = "senior business partner talent acquisition experience"

for name, model in models.items():
    method = name.split('_')[-1]
    
    # Kriteri preprocess işlemine sokuyoruz
    tokens = preprocess(custom_query, method)
    
    # OOV (Sözlük dışı kelimeler) durumları için Sıfır Vektör (Zero Vector) koruması
    q_vec = get_vec(model, tokens).reshape(1, -1)
    
    scores = []
    for _, row in df.iterrows():
        d_vec = get_vec(model, preprocess(row['job_description'], method)).reshape(1, -1)
        scores.append(cosine_similarity(q_vec, d_vec)[0][0])
    
    df['score'] = scores
    
    # Dışarıdan sorgu yaptığımız için kendisiyle eşleşme riski yok, direkt ilk 5 ilanı alıyoruz
    top5 = df.sort_values('score', ascending=False).iloc[0:5]
    
    # Jaccard hesabı için döküman ID listesini sakla
    all_results[name] = top5['doc_id'].tolist()
    
    # Raporlama formatına (final_similarity_results) uygun satırları ekle
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        final_similarity_rows.append({
            'Model_Name': f"word2vec_{name.lower()}.model",
            'Rank': rank,
            'Doc_Index': str(row['doc_id']).replace("doc_", ""), # Sadece sayısal indeks kalacak şekilde temizleme
            'Similarity_Score': round(row['score'], 4)
        })

# DataFrame'e dönüştür ve kaydet
df_final_similarity = pd.DataFrame(final_similarity_rows)
df_final_similarity.to_csv('final_similarity_results (1).csv', index=False, encoding='utf-8')
print(" -> [OK] final_similarity_results (1).csv başarıyla oluşturuldu.")


# ==============================================================================
# --- 5. COSINE EVALUATION SUMMARY TABLOSUNUN OLUŞTURULMASI ---
# ==============================================================================
cosine_summary_rows = []

for name in models.keys():
    model_full_name = f"word2vec_{name.lower()}.model"
    
    # Bu modele ait skorları filtrele ve listeye al
    model_scores = df_final_similarity[df_final_similarity['Model_Name'] == model_full_name]['Similarity_Score'].tolist()
    
    # Olası eksik veri durumunda listenin 5 elemana tamamlanmasını garanti et
    while len(model_scores) < 5:
        model_scores.append(0.0)
        
    avg_score = round(np.mean(model_scores), 4)
    
    cosine_summary_rows.append({
        'Model_Adı': model_full_name,
        'Skor_1': model_scores[0],
        'Skor_2': model_scores[1],
        'Skor_3': model_scores[2],
        'Skor_4': model_scores[3],
        'Skor_5': model_scores[4],
        'Ortalama_Skor': avg_score
    })

df_cosine_summary = pd.DataFrame(cosine_summary_rows)
df_cosine_summary.to_csv('cosine_evaluation_summary.csv', index=False, encoding='utf-8')
print(" -> [OK] cosine_evaluation_summary.csv başarıyla oluşturuldu.")


# ==============================================================================
# --- 6. JACCARD MATRİSİ VE ISI HARİTASI ÇIKTILARI ---
# ==============================================================================
model_names = list(all_results.keys())
jaccard_matrix = np.zeros((len(model_names), len(model_names)))

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        set1, set2 = set(all_results[m1]), set(all_results[m2])
        if len(set1 | set2) > 0:
            jaccard_matrix[i][j] = len(set1 & set2) / len(set1 | set2)
        else:
            jaccard_matrix[i][j] = 0.0

# Jaccard Matrisini CSV olarak kaydet (Raporlama şablonuna tam uygun model isimleriyle)
report_model_names = [f"word2vec_{name.lower()}.model" for name in model_names]
df_jaccard = pd.DataFrame(jaccard_matrix, index=report_model_names, columns=report_model_names)
df_jaccard.to_csv('jaccard_similarity_matrix.csv', index_label='', encoding='utf-8')
print(" -> [OK] jaccard_similarity_matrix.csv başarıyla oluşturuldu.")

# Görsel ısı haritasını rapor klasörüne kaydetme
plt.figure(figsize=(14, 10))
sns.heatmap(pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Modeller Arası Jaccard Benzerlik Matrisi (Sıralama Kararlılığı Analizi)")
plt.tight_layout()
plt.savefig(f"{output_folder}/jaccard_heatmap.png")
print(" -> [OK] Jaccard Isı Haritası grafiği kaydedildi.")


# ==============================================================================
# --- 7. SUMMARY TABLOSUNUN OLUŞTURULMASI ---
# ==============================================================================
summary_data = {
    'Baslik': ['Model Sayısı', 'Veri Seti', 'Kullanılan Yöntemler', 'Vector Boyutları', 'Window Size'],
    'Deger': ['16', 'all_job_post.csv', 'Word2Vec (CBOW ve Skip-Gram)', '100 ve 300', '2 ve 4']
}

pd.DataFrame(summary_data).to_csv('summary.csv', index=False, encoding='utf-8')
print(" -> [OK] summary.csv başarıyla oluşturuldu.")

print("\n[TEBRİKLER] Pipeline sorunsuz tamamlandı. Tüm CSV dosyaları Word raporu oluşturmaya hazır!")
