from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

# --- YARDIMCI FONKSİYONLAR (Hata almamak için bunlar şart) ---
def create_element(name):
    return OxmlElement(name)

def create_attribute(element, name, value):
    element.set(qn(name), value)

def add_page_number(run):
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "PAGE"
    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'separate')
    fldChar3 = create_element('w:fldChar')
    create_attribute(fldChar3, 'w:fldCharType', 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

# Paragraf ve Başlık Yardımcıları
doc = Document() # Belge burada tanımlanıyor, en kritik nokta burası

def P(text, bold=False, size=12, align="left"):
    p = doc.add_paragraph(text)
    if bold: p.runs[0].bold = True
    p.runs[0].font.size = Pt(size)
    if align == "center": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def H(text, level):
    return doc.add_heading(text, level=level)

def bullet(text):
    return doc.add_paragraph(text, style='List Bullet')

# --- ANA DOKÜMAN OLUŞTURMA ---

# BÖLÜM 1: KAPAK
section_1 = doc.sections[0]
section_1.footer.is_linked_to_previous = False

P("YAPAY ZEKA DERSI", bold=True, size=16, align="center")
P("Dönem Projesi Ödev-2 Raporu", size=14, align="center")
doc.add_paragraph()
P("WORD2VEC MODELLERİ İLE", bold=True, size=18, align="center")
P("METİN BENZERLİĞİ HESAPLAMASI VE ÇOK KRİTERLİ DEĞERLENDİRME", bold=True, size=18, align="center")
doc.add_paragraph()
P("Proje Konusu: İnsan Kaynakları İş İlanları Metin Analizi", size=12, align="center")
doc.add_paragraph()
P("Hazırlayanlar", bold=True, size=13, align="center")
P("Emre", size=12, align="center")
P("Hüseyin", size=12, align="center")
doc.add_paragraph()
P("Teslim Tarihi: 15 Haziran 2026", size=11, align="center")
doc.add_page_break()

# ÖZET
H("Özet", 1)
P("Bu projede, iş ilanlarından oluşan veri setimiz üzerinde Word2Vec kelime gömme modellerini kullanarak dökümanlar arası anlamsal benzerlik hesaplamaları gerçekleştirdik.")
doc.add_page_break()

# BÖLÜM 2: ANA İÇERİK
doc.add_section()
section_2 = doc.sections[1]
section_2.footer.is_linked_to_previous = False
footer_para = section_2.footer.paragraphs[0]
footer_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
add_page_number(footer_para.add_run())

H("1. Giriş ve Akademik Amaç", 1)
P("Bu projenin temel akademik amacı, kelimelerin boyutsal uzaydaki semantik temsillerini inşa etmektir.")

H("3. Bulgular", 1)
file_heatmap = "jaccard_heatmap.png"
if os.path.exists(file_heatmap):
    doc.add_picture(file_heatmap, width=Inches(6.0))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P("Şekil 1: Modeller Arası Sıralama Tutarlılığı (Jaccard Isı Haritası)", size=9, align="center")

# KAYDETME
out_docx = "Emre_Huseyin_Proje_Raporu.docx"
doc.save(out_docx)
print(f"Başarılı! Dosya kaydedildi: {out_docx}")
