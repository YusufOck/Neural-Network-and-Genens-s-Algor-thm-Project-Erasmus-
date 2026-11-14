# Trainer Hardening v1 — Doğrulanmış Altyapı Sonucu

Tarih: 2026-11-14

## Bu adımın türü

Bu çalışma **eğitim altyapısı geliştirmesi ve smoke testidir**. Nihai ResNet18 deneyi, adil model
karşılaştırması veya raporlanabilir kanser sınıflandırma sonucu değildir.

## Eklenen korumalar ve çıktılar

- `src/train.py`, sabit split manifestini eğitim başlamadan önce SHA-256 ile doğrular. Kabul edilen
  değer `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171` olmalıdır.
- 26 leaf tahminleri deterministik leaf→parent eşlemesiyle sekiz dataset üst grubuna dönüştürülür.
- Leaf ve parent seviyelerinde accuracy, macro-F1, balanced accuracy, classification report ve
  confusion matrix ayrı kaydedilir.
- Her test görüntüsü için relative path, gerçek/tahmin leaf, gerçek/tahmin parent, iki doğruluk
  bayrağı, tahmin güveni ve bütün sınıf olasılıkları `test_predictions.csv` dosyasına yazılır.
- Tamamen dondurulmuş BatchNorm katmanları eğitim sırasında evaluation modunda tutulur; böylece
  dondurulmuş ImageNet omurgasının running statistics değerleri sessizce değişmez.
- Validation macro-F1 için varsayılan patience=4 erken durdurma eklendi. En iyi epoch ve en iyi
  validation macro-F1 test özetine yazılır.
- Train/validation loss, accuracy ve macro-F1 eğrileri `learning_curves.png` olarak üretilir.
- Checkpoint güvenli `weights_only=True` yüklemesiyle geri açılır.

## Doğrulama

- Altı yeni birim testi leaf→parent projeksiyonu, aynı parent içindeki yanlış leaf örneği,
  çelişkili taxonomy reddi, frozen/trainable BatchNorm davranışı, görüntü bazlı tahmin satırları,
  probability toplamları ve CSV/hash üretimini doğruladı.
- Tam proje test paketi: **186/186 geçti**.
- Yerel Python 3.10 ortamındaki eksik `scikit-learn` kuruldu ve sürüm `1.7.2` olarak doğrulandı.
- Son kodla SmallCNN, sekiz parent hedefi, sınıf başına 40 görüntü ve tek epoch kullanan smoke test
  başarıyla tamamlandı. 320 görüntü 224/48/48 olarak işlendi; 48 test tahmin satırı ve beklenen
  rapor/grafik dosyaları üretildi.

Smoke testte görülen `%50 accuracy` ve `0.4326 macro-F1` küçük, rastgele image-level örneğe aittir.
Bu değerler proje performansı olarak kullanılamaz. Smoke hedefi sekiz parent olduğu için bu koşuda
"target" ve "parent" metriklerinin aynı olması beklenir; asıl 26-leaf ResNet18 koşusunda iki seviye
farklı ve ayrı raporlanacaktır.

## Bir sonraki kontrollü adım

İlk tam baseline için pretrained ResNet18, 26 leaf hedefi ve frozen practical manifest kullanılacak.
Model seçimi yalnız validation macro-F1 ile yapılacak; test seti en iyi checkpoint yüklendikten sonra
bir kez değerlendirilecek. Bu tam eğitim ayrı kullanıcı onayı ve Colab/T4 çalışma komutu gerektirir.

