from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd
import os

def P(doc, text, bold=False, size=12, align="left"):
    p = doc.add_paragraph(text)
    if bold: p.runs[0].bold = True
    p.runs[0].font.size = Pt(size)
    if align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER

def add_csv_to_doc(doc, file_name):
    if os.path.exists(file_name):
        P(doc, f"Veri Tablosu: {file_name}", bold=True, size=13)
        try:
            df = pd.read_csv(file_name)
            # Sadece ilk 10 satırı alalım ki rapor çok uzamasın
            df = df.head(10)
            
            table = doc.add_table(rows=df.shape[0]+1, cols=df.shape[1])
            table.style = 'Table Grid'
            for j, col_name in enumerate(df.columns):
                table.cell(0, j).text = str(col_name)
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    table.cell(i+1, j).text = str(df.iloc[i, j])
            doc.add_paragraph()
        except Exception as e:
            P(doc, f"Dosya okunamadı: {e}")
    else:
        # Dosya yoksa hata verme, sadece bilgilendir
        print(f"Uyarı: {file_name} dosyası bulunamadı, bu bölüm atlanıyor.")

# Ana Belge
doc = Document()
doc.add_heading("Proje Detaylı Analiz Raporu", 0)

# Elimizde OLAN dosyaların listesi
files_to_add = [
    "stemmed.csv",
    "all_job_post.csv",
    "lemmatized.csv"
]

# Tabloları ekle
for f in files_to_add:
    add_csv_to_doc(doc, f)

# Resmi ekle
if os.path.exists("jaccard_heatmap.png"):
    P(doc, "Jaccard Benzerlik Isı Haritası", bold=True, size=13)
    doc.add_picture("jaccard_heatmap.png", width=Inches(6.0))

# Kaydet
out_file = "Detayli_Proje_Raporu.docx"
doc.save(out_file)
print(f"Başarılı! Rapor oluşturuldu: {out_file}")
