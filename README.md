CV-İlan Eşleşmesi: Word2Vec Tabanlı Metin Benzerliği Analizi
Bu proje, Doğal Dil İşleme (NLP) yöntemleri ve Word2Vec kelime gömme (word embedding) algoritması kullanılarak geliştirilmiş bir İnsan Kaynakları (İK) projesidir. Projenin amacı, örnek bir iş ilanı metni (sorgu) ile veri setindeki aday özgeçmişleri (CV'ler) arasındaki anlamsal benzerliği hesaplayarak en uygun adayları sıralamaktır.

Proje kapsamında 16 farklı Word2Vec model varyasyonu (Lemmatized/Stemmed, CBOW/Skip-Gram, farklı pencere ve vektör boyutları) eğitilmiş ve hiperparametrelerin anlamsal başarı üzerindeki etkileri Kosinüs Benzerliği (Cosine Similarity) ve Jaccard Matrisi ile analiz edilmiştir.

🚀 Proje Yapısı ve Dosyalar
data/ : Ham ve ön işlenmiş veri setleri (lemmatized.csv, stemmed.csv).

models/ : Eğitilmiş 16 adet Word2Vec .model dosyası.

notebooks/ : Model eğitimi, benzerlik hesaplamaları ve görselleştirme kodlarını içeren Jupyter Notebook dosyaları.

reports/ : Isı haritaları, kosinüs özet tabloları ve performans analizleri.

README.md : Proje tanıtımı ve kurulum kılavuzu.

🛠️ Model Hiperparametreleri
Projede kullanılan kombinasyonlar şu şekildedir:

Veri Ön İşleme (Preprocessing): Lemmatization (Kelimelerin sözlük köklerine indirilmesi) vs. Stemming (Eklerin katı kurallarla kırpılması)

Algoritmik Mimari: CBOW (Continuous Bag-of-Words) ve Skip-Gram

Pencere Genişliği (Window Size): 2 ve 4

Vektör Boyutu (Vector Dimension): 100 ve 300

📊 Öne Çıkan Bulgular ve Değerlendirme
1. Ön İşleme Kırılımı (Lemmatization vs Stemming)
Lemmatized Modeller: 'management' terimi için managing, coordination, planning, delegation gibi anlamsal olarak yüksek tutarlılığa sahip, profesyonel İK terminolojisine uygun sonuçlar üretmiştir. Ortalama Kosinüs Benzerlik skorları kararlı bir şekilde 0.95+ bandında seyretmiştir.

Stemmed Modeller: Porter/Snowball benzeri kök kırpma işlemleri kelimelerin anlamsal bütünlüğünü bozduğu için management terimine en yakın kelimeler olarak relax, stuff, lush gibi bağlam dışı sonuçlar üretmiştir. Matematiksel olarak bazı durumlarda 0.95+ skorlara ulaşsalar da, bu durum tamamen anlamsal gürültüden (semantic noise) kaynaklanmaktadır.

2. En Başarılı Model
Projenin kavramsal derinliği en yüksek ve İK bağlamını en iyi koruyan modeli word2vec_lemmatized_cbow_win4_dim300.model olarak belirlenmiştir. Girdi metnindeki liderlik ve yönetim vizyonunu en optimize şekilde aday havuzuna yansıtabilmiştir.

3. Jaccard Matrisi (Sıralama Kararlılığı) Analizi
Modellerin getirdiği ilk 5 döküman üzerinden yapılan Jaccard analizinde, lemmatized modeller kendi aralarında yüksek tutarlılık sergilerken, stemmed modellerle karşılaştırıldıklarında benzerlik sıfıra yaklaşmıştır. Bu, veri ön işleme stratejisinin mimari seçiminden daha baskın bir yönlendirici olduğunu kanıtlamaktadır.

lemmatized_cbow_win2_dim100 ile win4_dim100 ve win2_dim300 modelleri arasında 1.00 tam Jaccard örtüşmesi bulunmuştur. Bu durum, CBOW mimarisinin parametre dalgalanmalarına karşı son derece kararlı (robust) bir duruş sergilediğini doğrulamaktadır.

💻 Kurulum ve Çalıştırma
Gereksinimler
Projenin çalışması için bilgisayarınızda Python 3.8+ yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için:

Bash
pip install gensim pandas numpy scikit-learn matplotlib seaborn
Projeyi Klonlayın
Bash
git clone https://github.com/[Kullanici_Adiniz]/CV-Job-Matching-NLP.git
cd CV-Job-Matching-NLP
Çalıştırma
Modelleri test etmek ve benzerlik analizini yeniden üretmek için notebooks/ dizinindeki ana Jupyter dosyasını çalıştırabilirsiniz:

Bash
jupyter notebook notebooks/main_analysis.ipynb
🛡️ Savunma Mekanizması (Zero Vector)
Sistem, sorgu veya doküman içeriklerinde tamamen bilinmeyen kelimeler (Out-of-Vocabulary) yer aldığında ZeroDivisionError veya NaN hataları oluşmasını engellemek amacıyla Sıfır Vektörü (Zero Vector) entegrasyonuna sahiptir. Bu sayede çalışma esnasında matematiksel güvenli alan korunmaktadır.
