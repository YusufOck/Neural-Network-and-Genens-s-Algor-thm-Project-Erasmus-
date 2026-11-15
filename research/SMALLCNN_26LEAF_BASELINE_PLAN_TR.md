# SmallCNN 26-Leaf Baseline Planı

Tarih: 2026-11-15

Durum: Deney sözleşmesi Mehmet tarafından onaylandı; kilitli/resumable Colab notebook'u hazır ve
196/196 proje testiyle doğrulandı. Eğitim henüz başlamadı.

Deney türü: Aynı sabit split üzerinde sıfırdan öğrenen model karşılaştırma baseline'ı.

## Hipotez ve amaç

İlk tam deneyde ImageNet-pretrained ve omurgası dondurulmuş ResNet18, 26 leaf sınıfında
`0,912455` test macro-F1 elde etti. SmallCNN deneyi, hazır ImageNet özellikleri kullanmadan aynı
görüntüler ve aynı değerlendirme protokolü üzerinde sıfırdan öğrenen küçük bir CNN'in seviyesini
ölçecektir. Amaç SmallCNN'in ResNet18'i geçmesini zorunlu tutmak değil; transfer learning'in
katkısını adil ve tekrar üretilebilir bir karşılaştırmayla görünür kılmaktır.

## Dondurulmuş deney sözleşmesi

- Hedef: 26 leaf etiketi; aynı leaf tahminlerinden sekiz parent veri-seti grubu türetilecek.
- Model: `SmallCNN`; üç convolution/BatchNorm/ReLU bloğu, iki max-pooling katmanı,
  adaptive average pooling, `0.35` dropout ve 26 çıkışlı doğrusal katman.
- Yaklaşık eğitilebilir parametre: 97.050; bütün ağırlıklar rastgele başlangıçtan öğrenilecek.
- Pretrained: kapalı; fine-tuning: uygulanamaz/kapalı.
- Split: `research/splits/practical_leakage_reduced_split_v1.csv`.
- Split SHA-256: `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Örnek sayıları: 91.000 train, 19.500 validation, 19.500 test; her leaf testte 750 görüntü.
- Girdi: `224x224`, ImageNet normalization ve yalnız train tarafında minimal augmentation.
- Optimizasyon: AdamW; learning rate `0.001`; weight decay `0.0001`.
- Batch size: `32`; seed: `42`.
- Bütçe: en fazla 25 epoch; validation macro-F1 beş epoch iyileşmezse erken durdurma.
- Checkpoint seçimi: yalnız validation macro-F1.
- Test: en iyi validation checkpoint'i seçildikten sonra yalnız nihai raporlama için açılacak.
- Kalıcı Colab output: `MyDrive/NNGA_OUTPUTS/smallcnn_26leaf_baseline_v1`.

Bu ayarlar ResNet18 test hatalarına göre seçilmemiştir. Test seti SmallCNN hiperparametresi veya
checkpoint'i seçmek için kullanılmayacaktır.

## Beklenen çıktılar

- Leaf ve türetilmiş parent accuracy, macro-F1 ve balanced accuracy.
- Sınıf başına precision, recall ve F1.
- Leaf ve parent confusion matrix.
- Train/validation öğrenme eğrileri.
- Tam 19.500 test görüntüsü için yol, gerçek/tahmin leaf-parent ve olasılık kayıtları.
- En iyi model, son-epoch resume checkpoint'i, frozen split kopyası ve çalışma konfigürasyonu.
- Büyük checkpoint'leri içermeyen, hash manifestli tek review ZIP paketi.

## Yorumlama kuralı

SmallCNN'in düşük veya yüksek çıkması tek başına başarısızlık/başarı kararı değildir. Aynı split ve
metriklerle ResNet18 head-only sonucuna göre farkı ölçülecek. Sekiz-parent sonucu klinik kanser
tanısı, kanser var/yok veya evre doğruluğu olarak yorumlanmayacaktır.

