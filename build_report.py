def create_attribute(element, name, value):
    element.set(ns.qn(name), value)

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

# ======================================================================================
# BÖLÜM 1: KAPAK VE ÖZET (SAYFA NUMARASI YOK)
# ======================================================================================
# İlk bölümün alt bilgisindeki sayfa numarasını kaldırıyoruz
section_1 = doc.sections[0]
section_1.footer.is_linked_to_previous = False
section_1.footer.paragraphs[0].text = ""

P("YAPAY ZEKA DERSI", bold=True, size=16, align="center")
P("Dönem Projesi Ödev-2 Raporu", size=14, align="center")
doc.add_paragraph()
P("WORD2VEC MODELLERİ İLE", bold=True, size=18, align="center")
P("METİN BENZERLİĞİ HESAPLAMASI VE ÇOK KRİTERLİ DEĞERLENDİRME", bold=True, size=18, align="center")
doc.add_paragraph()
P("Proje Konusu: İnsan Kaynakları İş İlanları Metin Analizi", size=12, align="center")
doc.add_paragraph()
doc.add_paragraph()

P("Hazırlayanlar", bold=True, size=13, align="center")
P("Emre", size=12, align="center")
P("Hüseyin", size=12, align="center")
doc.add_paragraph()

P("Teslim Tarihi: 15 Haziran 2026", size=11, align="center")
doc.add_page_break()

# ----------------- ÖZET (SAYFA 2) -----------------
H("Özet", 1)
P("Bu projede, iş ilanlarından oluşan veri setimiz üzerinde Word2Vec kelime gömme modellerini kullanarak "
  "dökümanlar arası anlamsal benzerlik hesaplamaları gerçekleştirdik. Çalışmamızın temel amacı, farklı "
  "hiperparametrelerin (CBOW/Skip-Gram, pencere genişliği, boyut) model başarısı üzerindeki etkisini "
  "ampirik olarak test etmektir. Vize aşamasında aldığımız geri bildirimler doğrultusunda, model seçimlerimizi "
  "sadece matematiksel kosinüs skorlarına değil, aynı zamanda insani bir değerlendirme sürecine dayandırdık. "
  "Elde ettiğimiz bulgular, teknik terim yoğunluklu metinlerde lemmatization ve Skip-Gram mimarisinin, "
  "kelime köklerini koruma ve anlamsal derinliği yakalama konusunda çok daha kararlı sonuçlar verdiğini "
  "ortaya koymaktadır.")

doc.add_paragraph()
H("Vize Değerlendirmesi ve Geri Bildirim Doğrultusunda İyileştirmeler", 2)
bullet("İlk rapordaki muğlak 'yerel olarak sağlanan' ifadesi kaldırılarak, 1167 adet dökümanlık 4.68 MB boyutundaki all_job_post.csv ana veri setinin kullanıldığı netleştirilmiştir.")
bullet("Stemming ve Lemmatization analizlerinin eksikliği giderilmiş, veri setinin küçük olması durumunda bu işlemlerin etkileri 'Niye Yaptık?' başlığı altında açıklanmıştır.")
bullet("Ara veri setleri (lemmatized.csv ve stemmed.csv) GitHub proje deposuna tam zamanlı olarak entegre edilmiştir.")
doc.add_page_break()

# ======================================================================================
# BÖLÜM 2: ANA İÇERİK (SAYFA NUMARALANDIRMASI BAŞLAR)
# ======================================================================================
doc.add_section()
section_2 = doc.sections[1]
section_2.footer.is_linked_to_previous = False
footer_para = section_2.footer.paragraphs[0]
footer_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
add_page_number(footer_para.add_run())

H("1. Giriş ve Akademik Amaç", 1)
P("Bu projenin temel akademik amacı, kurumsal insan kaynakları dünyasındaki iş tanımlarını ve "
  "gereksinimlerini içeren bir metin havuzunda, kelimelerin boyutsal uzaydaki semantik temsillerini "
  "inşa etmektir. Doğal Dil İşleme (NLP) literatüründe kelimelerin bağlamsal yakınlıklarını yakalamak, "
  "salt kelime eşleşmelerinin ötesine geçerek dökümanlar arası derin anlamsal benzerlikleri ölçmeyi "
  "mümkün kılar.")

H("2. Yöntem ve Deneysel Tasarım", 1)
P("Deneysel süreçte, metinlerin temizlenmiş iki farklı morfolojik versiyonu (Lemma ve Stem) girdi olarak "
  "kabul edilmiş; Gensim kütüphanesi vasıtasıyla 2 farklı mimari (CBOW ve Skip-Gram), 2 farklı bağlam "
  "penceresi (Window=2 ve 4) ve 2 farklı vektör boyutu (Size=100 ve 300) kombinasyonu kullanılarak toplam "
  "16 adet özgün Word2Vec modeli eğitilmiştir.")

H("2.1 Döküman Vektörlerinin Çıkarılması ve Çalışma Zamanı Savunma Mekanizması", 2)
P("Eğitilen modeller kelime tabanlı olduğundan, döküman seviyesinde bir temsil elde etmek amacıyla "
  "metin içerisindeki tüm kelimelerin vektörlerinin aritmetik ortalaması alınarak 'Döküman Vektörü' oluşturulmuştur. "
  "Metin ön işleme adımlarında stop-words temizlenirken bazı çok kısa iş ilanlarının içerisindeki tüm kelimelerin "
  "model dağarcığının (vocabulary) dışında kalma riskine karşı, boş kalan dökümanlara otomatik olarak "
  "Sıfır Vektörü (Zero Vector) atanmış ve çalışma zamanı (runtime) çökmeleri engellenmiştir.")

H("3. Bulgular ve Deneysel Sonuçların Analizi", 1)
H("3.1 Kök Bulma Yöntemlerinin Kelime Dağarcığına Etkisi (Niye Yaptık?)", 2)
P("Stemming (gövdelendirme) işlemi, kelimelerin sonundaki ekleri katı kurallarla kestiği için teknik İngilizce "
  "terimlerin yapısını bozmaktadır (Örn: 'programming' -> 'progr'). Bu durum, veri setinin küçük olmasıyla "
  "birleştiğinde modelin anlamlı bağlar kurmasını engeller. Lemmatization ise kelime türünü korur. Bu analizi "
  "yapmamızın gayesi, morfolojik tercihlerin modelin anlamsal uzayını nasıl şekillendirdiğini ispatlamaktır.")

H("3.2 Matematiksel Skor ile İnsani Algı Arasındaki Çelişkinin Analizi", 2)
P("Çalışmamızın en kritik bilimsel çıktısı; nesnel (Kosinüs) skorlarda en yüksek başarıyı elde eden modelin (M7), "
  "öznel (Anlamsal) değerlendirmede en yüksek puanı alan modelle (M8) aynı olmamasıdır. CBOW mimarisi (M7), "
  "iş ilanlarındaki 'takım çalışmasına yatkın' gibi kurumsal kalıpları ezberlemiştir. Skip-Gram (M8) ise "
  "'Python' ve 'Machine Learning' gibi ayırt edici anahtar kelimelerin ağırlığını korumuş ve insani algıya en "
  "uygun eşleştirmeleri yapmıştır.")

H("3.3 Sıralama Tutarlılığı (Jaccard) Analizi", 2)
P("Modellerin getirdiği ilk 5 döküman kümelerinin örtüşme düzeyleri Jaccard matrisi üzerinden hesaplanmıştır.")

try:
    if os.path.exists(file_heatmap):
        doc.add_picture(file_heatmap, width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        P("Şekil 1: Modeller Arası Sıralama Tutarlılığı (Jaccard Isı Haritası)", size=9, align="center")
    else:
        P(f"[GÖRSEL BULUNAMADI: {file_heatmap}]", italic=True)
except Exception as e:
    pass

P("Isı haritası incelendiğinde, aynı mimari sınıfına (CBOW veya Skip-Gram) ait modellerin kendi içlerinde "
  "yüksek sıralama tutarlılığı gösterdiği tespit edilmiştir. Ancak CBOW ile Skip-Gram sınıfları "
  "karşılaştırıldığında iki mimarinin metin uzayını tamamen farklı prensiplerle organize ettiği kanıtlanmıştır.")

H("4. Sonuç ve GitHub Depo Mimarisi", 1)
P("Bu proje, literatürde sıkça karşılaşılan 'kosinüs benzerliği her zaman doğru sonucu verir' yanılgısını "
  "bizzat deneyimlememizi sağladı. İş ilanı eşleştirme görevlerinde Lemmatized, Skip-Gram, geniş bağlam "
  "pencereli (window=4) ve yüksek boyutlu (dim=300) model yapısının (M8) en başarılı sonuçları ürettiği "
  "tespit edilmiştir.")

P("Çalışmanın doğrulanabilirliğini sağlamak adına GitHub repomuz şu şekilde yapılandırılmıştır:")
bullet("all_job_post.csv: Giriş ham veri seti.")
bullet("lemmatized.csv ve stemmed.csv: Kod tarafından üretilen ara veri setleri.")
bullet("odev2_hazirla.py: Word2Vec modellerini eğiten ana Python kodu.")
bullet("build_report.py: Bu rapor otomasyonunu sağlayan yan betik.")

out_docx = os.path.join(REP, "Emre_Huseyin_Proje_Raporu.docx")
doc.save(out_docx)
print(f"Başarılı! DOCX dosyası kaydedildi: {out_docx}")
