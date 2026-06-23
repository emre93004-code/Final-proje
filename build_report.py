import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- 1. AYARLAR ---
doc = Document()
doc.add_heading('Yapay Zeka Dersi - Ödev 2 Raporu', 0)

# Dosya yükleme yardımcısı
def load_csv(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    print(f"UYARI: {filename} bulunamadı!")
    return pd.DataFrame()

# --- 2. İÇERİK EKLEME FONKSİYONLARI ---
def add_table(df, title):
    doc.add_heading(title, level=2)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = 'Table Grid'
    hdr_cells = t.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = col
    
    for _, row in df.iterrows():
        row_cells = t.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)

# --- 3. RAPORU İNŞA ET ---
# Özet Tablosu
summary = load_csv("summary.csv")
if not summary.empty:
    add_table(summary, "Proje Özeti")

# Yöntem ve Detaylar
doc.add_heading("Yöntem", level=1)
doc.add_paragraph("Bu çalışmada 16 farklı Word2Vec modeli eğitilmiş, benzerlik hesaplamaları için Kosinüs ve Jaccard metrikleri kullanılmıştır.")

# Sonuçlar
doc.add_heading("Sonuçlar ve Değerlendirme", level=1)
add_table(load_csv("top5_per_model.csv"), "Modellerin İlk 5 Benzer Metin Çıktıları")
add_table(load_csv("cosine_eval.csv"), "Kosinüs Benzerliği Değerlendirmesi")

# Görsel (Heatmap)
if os.path.exists("jaccard_heatmap.png"):
    doc.add_heading("Jaccard Benzerlik Matrisi", level=2)
    doc.add_picture("jaccard_heatmap.png", width=Inches(6.0))

# Yorumlama Başlığı (Boş bırakıyoruz ki sen doldurabilesin)
doc.add_heading("Tartışma ve Yorumlama", level=1)
doc.add_paragraph("Buraya hangi modelin neden daha başarılı olduğunu, pencere boyutu ve vektör boyutunun performansa etkisini kendi cümlelerinle yazmalısın.")

# Sonuç ve Öneriler
doc.add_heading("Sonuç ve Öneriler", level=1)
doc.add_paragraph("Genel çıkarımlarınız ve gelecek çalışmalar için önerileriniz.")

# Kaydet
output_path = "Odev2_Raporu.docx"
doc.save(output_path)
print(f"Rapor başarıyla oluşturuldu: {output_path}")
