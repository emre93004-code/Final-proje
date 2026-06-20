# Word2Vec Dil Modelleri ile Metin Benzerliği Hesaplaması ve Çok Kriterli Performans Değerlendirmesi

## 📌 Yapay Zeka Dersi - Dönem Projesi Ödev-2 Raporu
**Hazırlayanlar:** Emre Yılmaz & Hüseyin  
**Proje Konusu:** İnsan Kaynakları Dünyasında Semantik Arama (CV - İş İlanı Eşleşmesi)  

---

## 📝 Proje Özeti ve Vize İyileştirmeleri
Bu çalışma, yapılandırılmamış metin verileri (İş İlanları) üzerinde farklı morfolojik yaklaşımların (Lemmatization ve Stemming) ve modern kelime gömme (**Word2Vec**) hiperparametrelerinin nihai döküman benzerlik sıralamaları üzerindeki etkisini ampirik olarak test etmektedir.

Vize aşamasındaki (Ödev-1) jüri geri bildirimleri doğrultusunda projede şu kritik iyileştirmeler yapılmıştır:
* **Amaç Tanımı:** Projenin İK dünyasındaki semantik eşleşme amacı netleştirilmiştir.
* **Morfolojik Analiz Ayrımı:** Kelimelerin sözlük kökleri (`lemmatized.csv`) ve morfolojik gövdeleri (`stemmed.csv`) için ayrı ayrı veri setleri üretilerek analizler derinleştirilmiştir.
* **Hata Savunma Mekanizması:** Giriş sorgusunda yer alan ancak modelin kelime haznesinde (vocabulary) bulunmayan kelimelerin sistemi çökertmesini engellemek adına **Sıfır Vektörü (Zero Vector)** koruması kod mimarisine entegre edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler ve Bağımlılıklar
Projenin çalıştırılabilmesi için bilgisayarınızda **Python 3.x** yüklü olmalıdır. Gerekli kütüphaneler:
* `gensim` (Word2Vec Model Eğitimi)
* `pandas` & `numpy` (Veri Manipülasyonu ve Matris İşlemleri)
* `scikit-learn` (Kosinüs Benzerliği Hesaplama)
* `seaborn` & `matplotlib` (Jaccard Isı Haritası Görselleştirme)
* `nltk` (Doğal Dil İşleme Ön İşlemleri)

---

## 🚀 Çalıştırma Talimatları

Hocamızın jüri değerlendirmesinde kodları tek komutla, kesintisiz ve teknik raporla birebir aynı çıktılar üretecek şekilde çalıştırabilmesi için sistem tam otomasyonlu bir terminal mimarisine indirgenmiştir.

### 1. Depoyu Klonlayın ve Klasöre Geçiş Yapın
```bash
git clone [https://github.com/emre93004-code/Final-proje.git](https://github.com/emre93004-code/Final-proje.git)
cd Final-proje
