# ResNet18 26-Leaf Baseline Sonucu

Tarih: 2026-11-15

Durum: Tam kapsamlı Colab/T4 koşusu tamamlandı ve indirilen artifact paketi yerelde doğrulandı.

Deney türü: Raporlanabilir ilk sabit-split baseline; klinik doğrulama değildir.

## Deney sözleşmesi

- Hedef: 26 leaf sınıfı; aynı tahminlerden sekiz parent veri-seti grubu türetildi.
- Model: ImageNet ağırlıklı ResNet18.
- Eğitim: Omurga dondurulmuş, yalnız 26 çıkışlı sınıflandırıcı başlığı eğitildi.
- Girdi: 224 x 224, ImageNet normalizasyonu, yalnız train tarafında minimal augmentation.
- Optimizasyon: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Batch: 32; seed: 42.
- Azami epoch: 15; validation macro-F1 için patience: 4.
- Split: `practical_leakage_reduced_split_v1.csv`, SHA-256
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Split büyüklükleri: 91.000 train, 19.500 validation, 19.500 test; her leaf testte 750 örnek.
- Checkpoint seçimi: yalnız validation macro-F1. Test sonuçları model seçimi bittikten sonra üretildi.

## Ana sonuç

| Seviye | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| 26 leaf | 0,913026 | 0,912455 | 0,913026 |
| Türetilmiş 8 parent grup | 0,997744 | 0,996756 | 0,996461 |

- En iyi validation checkpoint'i: epoch 9.
- En iyi validation macro-F1: 0,917727.
- Test loss: 0,216440.
- Tamamlanan epoch: 13. Epoch 9 sonrasında dört epoch iyileşme olmadığı için erken durdurma doğru
  biçimde çalıştı.
- Validation-test macro-F1 farkı yaklaşık 0,00527, yani 0,53 yüzde puandır.

## Leaf sınıf sonuçları

| Sınıf | Precision | Recall | F1 | Doğru / 750 |
|---|---:|---:|---:|---:|
| all_benign | 0,940 | 0,976 | 0,957 | 732 |
| all_early | 0,943 | 0,935 | 0,939 | 701 |
| all_pre | 0,984 | 0,956 | 0,970 | 717 |
| all_pro | 0,993 | 0,992 | 0,993 | 744 |
| brain_glioma | 0,965 | 0,931 | 0,948 | 698 |
| brain_menin | 0,913 | 0,867 | 0,889 | 650 |
| brain_tumor | 0,897 | 0,973 | 0,934 | 730 |
| breast_benign | 0,934 | 0,979 | 0,956 | 734 |
| breast_malignant | 0,983 | 0,949 | 0,966 | 712 |
| cervix_dyk | 0,895 | 0,912 | 0,904 | 684 |
| cervix_koc | 0,863 | 0,880 | 0,871 | 660 |
| cervix_mep | 0,944 | 0,881 | 0,912 | 661 |
| cervix_pab | 0,989 | 0,988 | 0,989 | 741 |
| cervix_sfi | 0,948 | 0,975 | 0,961 | 731 |
| colon_aca | 0,978 | 0,988 | 0,983 | 741 |
| colon_bnt | 0,997 | 0,993 | 0,995 | 745 |
| kidney_normal | 0,975 | 0,945 | 0,960 | 709 |
| kidney_tumor | 0,946 | 0,976 | 0,961 | 732 |
| lung_aca | 0,942 | 0,908 | 0,925 | 681 |
| lung_bnt | 0,989 | 0,999 | 0,994 | 749 |
| lung_scc | 0,910 | 0,944 | 0,927 | 708 |
| lymph_cll | 0,734 | 0,860 | 0,792 | 645 |
| lymph_fl | 0,756 | 0,789 | 0,772 | 592 |
| lymph_mcl | 0,803 | 0,631 | 0,706 | 473 |
| oral_normal | 0,809 | 0,689 | 0,744 | 517 |
| oral_scc | 0,736 | 0,823 | 0,777 | 617 |

## En önemli karışmalar

En büyük off-diagonal test hataları:

1. `oral_normal -> oral_scc`: 217.
2. `lymph_mcl -> lymph_cll`: 153.
3. `lymph_mcl -> lymph_fl`: 124.
4. `oral_scc -> oral_normal`: 121.
5. `lymph_fl -> lymph_cll`: 81.
6. `brain_menin -> brain_tumor`: 78.
7. `lymph_fl -> lymph_mcl`: 77.
8. `lymph_cll -> lymph_fl`: 66.
9. `lung_aca -> lung_scc`: 63.

Bu tablo, ortalama başarı yüksek olsa da lymphoma ve oral leaf ayrımlarının belirgin biçimde daha
zor olduğunu gösterir. Sonraki model/fine-tuning karşılaştırmalarında yalnız toplam accuracy değil,
özellikle bu sınıfların recall ve F1 değişimleri izlenmelidir.

## Parent sonucunun doğru yorumu

19.500 test görüntüsünün 19.456'sında parent grup doğru, 44'ünde yanlıştır. Parent düzeyindeki
yüksek 0,997744 accuracy şu anlama gelir: görüntünün hangi üst veri-seti/organ-modalite grubuna ait
olduğu çoğunlukla doğru bulunmuştur. Bu sonuç:

- kanser var/yok doğruluğu değildir;
- kanser evresi doğruluğu değildir;
- klinik tanı başarısı değildir.

Sekiz üst grup farklı modaliteler ve görsel kaynaklar içerdiği için model organ/modalite ipuçlarını
kolayca ayırabilir. Nitekim leaf düzeyinde `oral_normal` ile `oral_scc` veya lymphoma alt türleri
karışırken parent grup yine doğru sayılmaktadır.

## Eğitim eğrisi yorumu

- Validation macro-F1 epoch 9'da 0,917727 ile zirveye ulaştı.
- Epoch 10-13 arasında train metriği yavaşça artarken validation metriği artık iyileşmedi; patience
  4 bu nedenle doğru noktada durdurdu.
- Validation skorunun train skorundan yüksek olması tek başına hata değildir. Train metriği
  augmentation uygulanmış daha zor görüntüler üzerinde ve eğitim modunda ölçülür; validation temiz
  görüntüler üzerinde ölçülür. Omurganın dondurulmuş olması da bu farkı destekler.
- Eğriler ağır bir çöküş göstermiyor; yine de epoch 9 sonrası sınırlı overfitting/plato işareti var.

## Bağımsız artifact doğrulaması

İndirilen review paketi:

- Dosya: `resnet18_26leaf_baseline_v1_review_package.zip`
- Boyut: 9.117.915 byte.
- SHA-256: `e7e4e40a081af1fb2afdb261e4cb4169e51f91741322ad38717106651a79c24c`.

Yerel yeniden hesaplamalar:

- Paket içindeki bütün mevcut dosyalar `artifact_hashes.json` ile hash ve boyut bakımından eşleşti.
- Checkpoint dosyaları pakete eklenmedi; Drive'da hesaplanmış hash'leri kayda alındı.
- Manifest: tam 91.000/19.500/19.500 ve cross-split family sayısı 0.
- Tahmin: 19.500 satır, 19.500 benzersiz yol, test dışı yol 0, label uyuşmazlığı 0.
- Her leaf test desteği: tam 750.
- Olasılık anahtar/toplam/confidence uyuşmazlığı: 0.
- Tahminlerden yeniden hesaplanan leaf doğru sayısı: 17.804/19.500.
- Tahminlerden yeniden hesaplanan parent doğru sayısı: 19.456/19.500.
- Test accuracy değerleri JSON ile birebir aynı.

Kalıcı hafif kanıtlar `research/model_results/resnet18_26leaf_baseline_v1/` altındadır. Büyük
`best_model.pth`, `last_checkpoint.pth`, `test_predictions.csv` ve kopya split manifest Drive'da
tutulur; hash'leri `artifact_hashes.json` ile sürümlenir.

## Sınırlamalar ve sonraki kontrollü adım

- Split bütün bilinen exact/review ailelerini birlikte tutar; bilinmeyen bütün near-duplicate,
  patient veya series ilişkilerini garanti etmez.
- Veri seti kaynak/modalite confounding taşır.
- Sonuç yalnız bu veri seti ve bu sabit split için eğitim amaçlı model performansıdır.
- Tek koşu vardır; seed varyansı henüz ölçülmemiştir.
- Bu head-only baseline'dır; kontrollü fine-tuning veya başka mimari sonucu değildir.

Sıradaki deney otomatik başlamamalıdır. Önce Mehmet ile bu baseline'ın confusion matrix'i ve
macro-F1 yorumu öğrenme kontrolünden geçirilir. Sonra aynı manifest üzerinde önceden sözleşmesi
dondurulmuş bir karşılaştırma seçilir: tercihen ResNet18 kontrollü fine-tuning veya SmallCNN
from-scratch baseline. Test seti yeni hiperparametre kararlarında kullanılmamalı; seçimler
validation üzerinden yapılmalıdır.
