# -*- coding: utf-8 -*-
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches

# --- EN KRİTİK ADIM: Doküman nesnesini burada tanımlıyoruz ---
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
    return pd.DataFrame()

# Tablo oluşturma fonksiyonu
def df_to_table(df, title):
    # doc nesnesi burada global olarak çağrılıyor
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

# --- RAPORU İNŞA ET ---
doc.add_heading('Yapay Zeka Dersi - Ödev 2 Raporu', 0)

# Verileri çek ve tablolara ekle
summary = load_file("summary.csv")
df_to_table(summary, "Proje Özeti")

top5 = load_file("top5_per_model.csv")
df_to_table(top5, "Modellerin İlk 5 Benzer Metin Çıktıları")

cos = load_file("cosine_eval.csv")
df_to_table(cos, "Kosinüs Benzerliği Değerlendirmesi")

# Yorum bölümleri
doc.add_heading("Tartışma ve Yorumlama", level=1)
doc.add_paragraph("Pencere boyutu 2 olan modeller, yakın bağlamı daha iyi yakaladığı için daha yüksek benzerlik skorları üretmiştir. CBOW mimarisi, teknik terimlerin tahmininde daha stabil sonuçlar vermiştir.")

doc.add_heading("Sonuç ve Öneriler", level=1)
doc.add_paragraph("Word2Vec modelleri teknik metin benzerliklerinde başarılı sonuçlar vermiştir. Gelecek çalışmalarda Transformer tabanlı modellerin kullanılması başarıyı artırabilir.")

# Kaydet
doc.save(os.path.join(REP, "Odev2_Raporu.docx"))
print("Rapor başarıyla oluşturuldu: /rapor/Odev2_Raporu.docx")
