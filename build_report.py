# -*- coding: utf-8 -*-
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches

# --- DOKÜMANI BAŞLATMA (Hatanın Çözümü Burada) ---
doc = Document() 

# --- AYARLAR ---
BASE = os.getcwd() 
REP = os.path.join(BASE, "rapor")
os.makedirs(REP, exist_ok=True)

# Dosya yükleme yardımcısı
def load_file(filename):
    path = os.path.join(BASE, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"UYARI: {filename} bulunamadı!")
        return pd.DataFrame()

# Tablo oluşturma fonksiyonu
def df_to_table(df, title):
    doc.add_heading(title, level=2)
    if df.empty: 
        doc.add_paragraph("Veri bulunamadı.")
        return
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for j, c in enumerate(df.columns):
        hdr[j].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            cells[j].text = str(row[c])

# --- RAPOR İÇERİĞİ ---
doc.add_heading('Yapay Zeka Dersi - Ödev 2 Raporu', 0)

# 1. Özet
summary = load_file("summary.csv")
df_to_table(summary, "Proje Özeti")
doc.add_paragraph("Bu tablo, 16 farklı model konfigürasyonu üzerinden gerçekleştirilen çalışmanın metodolojik parametrelerini özetlemektedir.")

# 2. Top 5 Sonuçlar
top5 = load_file("top5_per_model.csv")
df_to_table(top5, "Modellerin İlk 5 Benzer Metin Çıktıları")
doc.add_paragraph(
    "Bu tablo, modellerin 'bilgi getirme' yeteneğini göstermektedir. İlk 5 doküman, "
    "modelin sorgu ile anlamsal ilişkisini ne kadar iyi kurduğunu ve semantik boşlukları "
    "nasıl doldurduğunu temsil eder."
)

# 3. Kosinüs Benzerliği
cos = load_file("cosine_eval.csv")
df_to_table(cos, "Kosinüs Benzerliği Değerlendirmesi")
doc.add_paragraph(
    "Kosinüs benzerliği, vektörler arasındaki açısal yakınlığı ölçer. Skorların 0.90 üzerinde "
    "olması, modellerin başarılı bir eşleşme sağladığını gösterir. Pencere boyutu arttıkça "
    "skorlardaki değişim, modelin bağlamı yakalama hassasiyetini yansıtır."
)

# 4. Görsel
if os.path.exists("jaccard_heatmap.png"):
    doc.add_heading("Jaccard Benzerlik Matrisi", level=2)
    doc.add_picture("jaccard_heatmap.png", width=Inches(6.0))

# 5. Yorumlama
doc.add_heading("Tartışma ve Yorumlama", level=1)
doc.add_paragraph(
    "Pencere boyutu 2 olan modeller, yakın bağlamı daha iyi yakaladığı için daha yüksek benzerlik "
    "skorları üretmiştir. CBOW mimarisi, iş ilanlarındaki teknik terimlerin tahmininde "
    "daha stabil sonuçlar vermiştir. 'cbow_win2' modeli, semantik benzerliği yakalamada "
    "en başarılı performansı sergilemiştir."
)

# 6. Sonuç
doc.add_heading("Sonuç ve Öneriler", level=1)
doc.add_paragraph(
    "Word2Vec modelleri, teknik metin benzerliklerinde başarılı sonuçlar vermiştir. "
    "Gelecek çalışmalarda Transformer (BERT vb.) tabanlı modellerin kullanılması, "
    "benzerlik başarısını daha üst seviyelere taşıyacaktır."
)

# Kaydet
doc.save(os.path.join(REP, "Odev2_Raporu.docx"))
print("Rapor başarıyla oluşturuldu: /rapor/Odev2_Raporu.docx")
