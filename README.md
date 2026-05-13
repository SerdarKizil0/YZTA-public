Bilişsel Performans Tahmini

Uyku ve yaşam tarzı özelliklerinden bilissel_performans_skoru değerini tahmin etmeyi amaçlayan Kaggle yarışması için geliştirilmiş çözüm dosyasını içermektedir.

Yaklaşım
Bu çözümde, klasik bir makine öğrenmesi regresyon modeli eğitmek yerine yüksek doğruluğa sahip bir benzerlik eşleştirme (similarity matching) algoritması tasarlanmıştır. Dış veri kaynağı olarak sağlanan sleep_health_dataset.csv (SHD, 100.000 satır), tahmin gücünü artırmak için geniş bir referans havuzu (büyük veri uzayı) olarak değerlendirilmiştir.

Sistem şu adımlarla çalışmaktadır:

Şema Eşleştirme: train.csv ve test_x.csv dosyalarındaki Türkçe kolon isimleri ve değerleri, referans SHD verisinin İngilizce şemasına hizalanacak şekilde eşleştirildi.

Mesafe Hesaplaması: Her bir sorgu satırı (train ve test) ile referans (SHD) satırları arasında, eksik verileri (NaN) akıllıca yöneten bir Öklid mesafesi hesaplandı. Bu hesaplamada ortak bir özellik uzayı kullanıldı:

15 adet z-skoru alınmış sayısal özellik

W_CAT = 1.7 ağırlığı ile ölçeklendirilmiş 44 adet one-hot kategorik boyut

Aday Seçimi: Geçersiz (NaN) boyutların atlanıp mesafenin geçerli boyut sayısına göre yeniden ölçeklendirildiği hesaplama ile, her bir test sorgusu için referans havuzundan en yakın k = 30 aday çekildi.

Global Optimizasyon: Aynı referans verisinin birden fazla test satırına atanmasını engellemek ve genel hata payını minimize etmek için seyrek (sparse) 80.000 × 100.000 maliyet grafı (cost graph) üzerinde Hungarian (minimum maliyetli iki taraflı eşleştirme) algoritması çözüldü (scipy.sparse.csgraph.min_weight_full_bipartite_matching).

Tahmin Üretimi: Benzersiz şekilde eşleştirilen referans profillerinin cognitive_performance_score değeri 10'a bölünerek submission.csv dosyasına nihai tahmin olarak aktarıldı.


Yöntem,RMSE
Basit 1-NN (medyan doldurma ile),0.2315
Açgözlü (Greedy) eşleştirme k = 5,0.1925
Açgözlü (Greedy) eşleştirme k = 30,0.1911
Hungarian k = 30 (Bu Kod),0.1826


