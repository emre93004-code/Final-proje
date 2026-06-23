# ==============================================================================
# --- 6. RAPORLAMA SCRIPTI İÇİN UYUMLU VERİ SETLERİNİ OLUŞTURMA (YENİ KISIM) ---
# ==============================================================================
print("\nRaporlama scripti için uyumlu CSV dosyaları üretiliyor...")

# A. summary.csv Oluşturulması
summary_data = {
    'Baslik': ['Model Sayısı', 'Veri Seti', 'Kullanılan Yöntemler', 'Vector Boyutları', 'Window Size'],
    'Deger': ['16', 'all_job_post.csv', 'Word2Vec (CBOW ve Skip-Gram)', '100 ve 300', '2 ve 4']
}
pd.DataFrame(summary_data).to_csv('summary.csv', index=False, encoding='utf-8')
print(" -> [OK] summary.csv başarıyla oluşturuldu.")


# B. final_similarity_results (1).csv Oluşturulması (Tüm Modellerin Ayrıntılı İlk 5 Doküman Çıktısı)
final_similarity_rows = []
for name, model in models.items():
    method = name.split('_')[-1]
    tokens = preprocess(df['job_description'].iloc[query_idx], method)
    q_vec = get_vec(model, tokens).reshape(1, -1)
    
    scores = []
    for _, row in df.iterrows():
        d_vec = get_vec(model, preprocess(row['job_description'], method)).reshape(1, -1)
        scores.append(cosine_similarity(q_vec, d_vec)[0][0])
    
    df['score'] = scores
    # Kendisi hariç en yakın ilk 5 dokümanı filtrele
    top5_docs = df.sort_values('score', ascending=False).iloc[1:6]
    
    for rank, (_, row) in enumerate(top5_docs.iterrows(), 1):
        final_similarity_rows.append({
            'Model_Name': f"word2vec_{name.lower()}.model",
            'Rank': rank,
            'Doc_Index': row['doc_id'].replace("doc_", ""), # Raporlama formatına uygun nümerik indeks
            'Similarity_Score': round(row['score'], 4)
        })

df_final_similarity = pd.DataFrame(final_similarity_rows)
df_final_similarity.to_csv('final_similarity_results (1).csv', index=False, encoding='utf-8')
print(" -> [OK] final_similarity_results (1).csv başarıyla oluşturuldu.")


# C. cosine_evaluation_summary.csv Oluşturulması (Modellerin Özet Tablosu)
cosine_summary_rows = []
for name in models.keys():
    # Bu modele ait skorları filtrele
    model_full_name = f"word2vec_{name.lower()}.model"
    model_scores = df_final_similarity[df_final_similarity['Model_Name'] == model_full_name]['Similarity_Score'].tolist()
    
    # 5 skorun tam listelenmesini garanti altına al
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


# D. jaccard_similarity_matrix.csv Oluşturulması
# Model isimlerini raporlama standart formatına çevirelim
report_model_names = [f"word2vec_{name.lower()}.model" for name in model_names]
df_jaccard = pd.DataFrame(jaccard_matrix, index=report_model_names, columns=report_model_names)
df_jaccard.to_csv('jaccard_similarity_matrix.csv', index_encoding='utf-8')
print(" -> [OK] jaccard_similarity_matrix.csv başarıyla oluşturuldu.")

print("\nTüm veri köprüleri hazır! Rapor oluşturma scriptini (Word builder) doğrudan çalıştırabilirsin.")
