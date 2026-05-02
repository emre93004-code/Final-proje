# İş İlanları Veri Seti Üzerinde NLP ve Zipf Yasası Analizi

Bu proje, yapılandırılmamış metin verilerinin (İş İlanları) Doğal Dil İşleme (NLP) teknikleri kullanılarak temizlenmesini ve dillerin istatistiksel dağılımını açıklayan Zipf Yasası'nın bu veriler üzerindeki geçerliliğinin analiz edilmesini içermektedir.

## 📝 Proje Özeti
* **Amaç:** İş ilanlarından oluşan geniş bir korpus üzerinde metin ön işleme adımlarını uygulamak ve kelime frekanslarının teorik dağılım ile uyumunu test etmek.[cite: 1]
* **Veri Seti:** LinkedIn/Kaggle üzerinden alınan iş ilanları (Job Postings) veri seti kullanılmıştır.[cite: 1, 2]
* **Yöntem:** NLTK kütüphanesi ile metin temizleme (Lowercasing, Stop-words removal, Lemmatization) ve Matplotlib ile log-log ölçeğinde frekans analizi yapılmıştır.[cite: 1]

## 🚀 Bulgular ve Sonuç
* **İstatistiksel Uyumluluk:** Analiz sonucunda elde edilen log-log grafiği, kelime kullanım sıklıklarının Zipf Yasası ile yüksek oranda örtüştüğünü kanıtlamıştır.[cite: 1]
* **Sektörel Çıkarım:** İş ilanları özelinde yapılan analizde; *experience, management, skills* gibi sektörel terimlerin frekans dağılımında baskın olduğu gözlemlenmiştir.[cite: 1]
* **Gelecek Çalışmalar:** Temizlenen veri seti, makine öğrenmesi tabanlı metin sınıflandırma ve anahtar kelime çıkarma modelleri için hazır hale getirilmiştir.[cite: 1]

## 🛠️ Kullanılan Teknolojiler
* Python 3.x
* Pandas (Veri manipülasyonu)
* NLTK (Doğal Dil İşleme)
* Matplotlib & Seaborn (Görselleştirme)

## 📦 Kurulum
Projeyi yerelinizde çalıştırmak için:
1. Depoyu klonlayın: `git clone https://github.com/kullaniciadi/repo-adi.git`
2. Gereksinimleri yükleyin: `pip install -r requirements.txt`
3. `NLP_Metin_On_Isleme_Final.ipynb` dosyasını çalıştırın.