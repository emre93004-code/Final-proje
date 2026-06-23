# -*- coding: utf-8 -*-
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# --- AYARLAR ---
BASE = os.getcwd() # Colab'da bulunduğun ana dizin (/content/)
REP = os.path.join(BASE, "rapor")
os.makedirs(REP, exist_ok=True)

# Dosyaları BASE (yani /content/) içinde ara
def load_file(filename):
    path = os.path.join(BASE, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"UYARI: {filename} dosyası bulunamadı, kontrol et!")
        return pd.DataFrame()

# Dosyaları yükle
cos = load_file("cosine_eval.csv")
sem = load_file("semantic_eval.csv")
sw = load_file("similar_words.csv")
top5 = load_file("top5_per_model.csv")
jac = load_file("jaccard_matrix.csv") if os.path.exists("jaccard_matrix.csv") else pd.DataFrame()
summary = load_file("summary.csv")

if os.path.exists("query.txt"):
    query_txt = open("query.txt", encoding="utf-8").read().strip().splitlines()
else:
    query_txt = ["Girdi metni bulunamadı."]

# ... (Diğer fonksiyonların aynı kalacak) ...
doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def H(text, level=1): return doc.add_heading(text, level=level)

def P(text, bold=False, italic=False, size=11, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    if align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def bullet(text): doc.add_paragraph(text, style="List Bullet")

def df_to_table(df, header_color="2E5496", col_widths=None, font_size=8):
    if df.empty: return
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for j, c in enumerate(df.columns):
        hdr[j].text = str(c)
        for pr in hdr[j].paragraphs:
            for rn in pr.runs: rn.bold = True; rn.font.size = Pt(font_size)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, c in enumerate(df.columns):
            cells[j].text = str(row[c])
            for pr in cells[j].paragraphs:
                for rn in pr.runs: rn.font.size = Pt(font_size)
    return t

# --- KAPAK BİLGİLERİNİ BURADAN DÜZENLE ---
P("YAPAY ZEKA DERSI", bold=True, size=14, align="center")
P("Odev-2", size=12, align="center")
doc.add_paragraph()
P("EGITILEN WORD2VEC MODELLERI ILE", bold=True, size=18, align="center")
P("METIN BENZERLIGI HESAPLAMA VE DEGERLENDIRME", bold=True, size=18, align="center")
doc.add_paragraph()
# ... (Kapak ve diğer kısımlar senin orijinal kodundaki gibi kalabilir) ...

# En son kaydetme kısmını kontrol et:
out_docx = os.path.join(REP, "Odev2_Raporu.docx")
doc.save(out_docx)
print("Başarılı! Rapor kaydedildi:", out_docx)
