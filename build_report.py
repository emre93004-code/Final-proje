# -*- coding: utf-8 -*-
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches

# --- 1. DOKÜMANI BAŞLATMA (Hatanın Çözümü Burada) ---
# Kodun en başında bunu tanımlıyoruz ki sonraki satırlarda hata vermesin.
doc = Document()

# --- 2. AYARLAR ---
BASE = os.getcwd() 
REP = os.path.join(BASE, "rapor")
os.makedirs(REP, exist_ok=True)

# Dosya yükleme yardımcısı
def load_file(filename):
    path = os.path.join(BASE, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"UYARI: {filename} dosyası bulunamadı!")
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

# --- 3. RAPOR İÇERİĞİ ---
# Artık 'doc' tanımlı olduğu için bu komutlar çalışacaktır
doc.add_heading('Yapay Zeka Dersi - Ödev 2 Raporu', 0)

# Özet Tablosu
summary = load_file("summary.csv")
df_to_table(summary, "Proje Özeti")

# 4. Sonuçlar ve Değerlendirme
top5 = load_file("top5_per_model.csv")
df_to_table(top5, "Modellerin İlk 5 Benzer Metin Çıktıları")

cos = load_file("cosine_eval.csv")
df_to_table(cos, "Kosinüs Benzerliği Değerlendirmesi")

# Görsel ekleme
if os.path.exists("jaccard_heatmap.png"):
    doc.add_heading("Jaccard Benzerlik Matrisi", level=2)
    doc.add_picture("jaccard_heatmap.png", width=Inches(6.0))

# Tartışma ve Sonuç
doc.add_heading("Tartışma ve Yorumlama", level=1)
doc.add_paragraph("Pencere boyutu 2 olan modeller, yakın bağlamı daha iyi yakaladığı için daha yüksek benzerlik skorları üretmiştir. CBOW mimarisi, teknik terimlerin tahmininde daha başarılı sonuçlar vermiştir.")

doc.add_heading("Sonuç ve Öneriler", level=1)
doc.add_paragraph("Word2Vec modelleri teknik metin benzerliklerinde başarılı olmuştur. Gelecek çalışmalarda Transformer tabanlı modellerin kullanımı, benzerlik başarısını artırabilir.")

# --- 4. KAYDETME ---
doc.save(os.path.join(REP, "Odev2_Raporu.docx"))
print("Rapor başarıyla oluşturuldu: /rapor/Odev2_Raporu.docx")
