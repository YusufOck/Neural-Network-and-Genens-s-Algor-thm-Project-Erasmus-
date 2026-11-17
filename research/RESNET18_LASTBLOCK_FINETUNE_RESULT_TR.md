# ResNet18 Last-Block Fine-Tuning Ablation Sonucu

Tarih: 2026-11-17

Durum: Tam kapsamlı Colab/T4 koşusu tamamlandı; indirilen review ZIP'i, artifact hash'leri,
19.500 test tahmini, frozen manifest üyeliği ve metrik aritmetiği yerelde bağımsız olarak
doğrulandı.

Deney türü: Aynı sabit benchmark üzerinde önceden sözleşmesi dondurulmuş kontrollü ablation.
Klinik doğrulama veya dokunulmamış yeni bir dış test değildir.

## Araştırma sorusu ve sözleşme

Bu deney, head-only ResNet18'in düşük kalmasının nedeninin ImageNet gövdesinin veri setine
uyarlanamaması olup olmadığını test eder.

- Başlangıç: validation ile seçilmiş önceki head-only ResNet18 epoch-9 `best_model.pth`.
- Init SHA-256:
  `20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177`.
- Eğitilen katmanlar: yalnız `layer4 + fc`.
- Eğitilebilir parametre: 8.407.066 / toplam 11.189.850.
- Eski optimizer/history taşınmadı; yeni AdamW koşusu başlatıldı.
- Learning rate: `0.0001`; weight decay: `0.0001`.
- Girdi: 224 x 224, ImageNet normalization, minimal train-only augmentation.
- Batch: 32; seed: 42; azami epoch: 12; validation macro-F1 patience: 4.
- Hedef: 26 leaf; aynı tahminlerden sekiz parent veri-seti grubu türetildi.
- Split: `practical_leakage_reduced_split_v1.csv`, SHA-256
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Split büyüklükleri: 91.000 train, 19.500 validation, 19.500 test; her leaf testte 750.
- Checkpoint seçimi yalnız validation macro-F1 ile yapıldı.

## Ana sonuç

| Seviye | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| 26 leaf | 0,996462 | 0,996462 | 0,996462 |
| Türetilmiş 8 parent grup | 0,999897 | 0,999858 | 0,999883 |

- En iyi checkpoint: epoch 11.
- En iyi validation macro-F1: `0,995948`.
- Test macro-F1: `0,996462`; validation farkı yalnız `+0,000514` veya yaklaşık `+0,0514`
  yüzde puandır.
- Test loss: `0,010893`.
- Tamamlanan epoch: 12. Patience tetiklenmeden dondurulmuş azami epoch'a ulaşıldı.
- Leaf doğru: 19.431 / 19.500; leaf hata: 69.
- Parent doğru: 19.498 / 19.500; parent hata: 2.
- Leaf hatalarının 67'sinde parent yine doğrudur.

## Leaf sınıf sonuçları

| Sınıf | Precision | Recall | F1 | Doğru / 750 |
|---|---:|---:|---:|---:|
| all_benign | 1,0000 | 1,0000 | 1,0000 | 750 |
| all_early | 1,0000 | 1,0000 | 1,0000 | 750 |
| all_pre | 1,0000 | 1,0000 | 1,0000 | 750 |
| all_pro | 1,0000 | 1,0000 | 1,0000 | 750 |
| brain_glioma | 1,0000 | 0,9947 | 0,9973 | 746 |
| brain_menin | 0,9907 | 0,9947 | 0,9927 | 746 |
| brain_tumor | 0,9947 | 0,9960 | 0,9953 | 747 |
| breast_benign | 0,9987 | 0,9893 | 0,9940 | 742 |
| breast_malignant | 0,9894 | 0,9987 | 0,9940 | 749 |
| cervix_dyk | 1,0000 | 1,0000 | 1,0000 | 750 |
| cervix_koc | 0,9973 | 1,0000 | 0,9987 | 750 |
| cervix_mep | 1,0000 | 0,9987 | 0,9993 | 749 |
| cervix_pab | 1,0000 | 1,0000 | 1,0000 | 750 |
| cervix_sfi | 1,0000 | 0,9987 | 0,9993 | 749 |
| colon_aca | 1,0000 | 1,0000 | 1,0000 | 750 |
| colon_bnt | 1,0000 | 1,0000 | 1,0000 | 750 |
| kidney_normal | 1,0000 | 1,0000 | 1,0000 | 750 |
| kidney_tumor | 1,0000 | 1,0000 | 1,0000 | 750 |
| lung_aca | 0,9960 | 0,9987 | 0,9973 | 749 |
| lung_bnt | 1,0000 | 1,0000 | 1,0000 | 750 |
| lung_scc | 0,9987 | 0,9947 | 0,9967 | 746 |
| lymph_cll | 0,9934 | 0,9960 | 0,9947 | 747 |
| lymph_fl | 1,0000 | 1,0000 | 1,0000 | 750 |
| lymph_mcl | 0,9960 | 0,9933 | 0,9947 | 745 |
| oral_normal | 0,9773 | 0,9760 | 0,9767 | 732 |
| oral_scc | 0,9761 | 0,9787 | 0,9774 | 734 |

En zayıf iki leaf hâlâ `oral_normal` ve `oral_scc` olsa da F1 değerleri sırasıyla 0,9767 ve
0,9774'tür. En sık hatalar:

1. `oral_normal -> oral_scc`: 18.
2. `oral_scc -> oral_normal`: 16.
3. `breast_benign -> breast_malignant`: 7.
4. `lymph_mcl -> lymph_cll`: 5.
5. `brain_glioma -> brain_menin`: 4.
6. `brain_menin -> brain_tumor`: 4.

Doğru tahminlerin ortalama confidence değeri 0,99768, yanlışların 0,79141'dir. Bununla birlikte
69 yanlışın 23'ü en az 0,90, yedisi en az 0,99 confidence taşır. Softmax confidence klinik
güvenilirlik veya kalibrasyon kanıtı değildir.

## Aynı test görüntülerinde adil karşılaştırma

Üç modelin de aynı 19.500 benzersiz test yolunu kullandığı doğrulandı.

| Ölçüt | Frozen-head ResNet18 | SmallCNN | Last-block ResNet18 |
|---|---:|---:|---:|
| Leaf accuracy | 0,913026 | 0,955846 | **0,996462** |
| Leaf macro-F1 | 0,912455 | 0,955916 | **0,996462** |
| Parent accuracy | 0,997744 | 0,999231 | **0,999897** |

Last-block fine-tuning farkları:

- Frozen-head ResNet18'e göre leaf accuracy `+0,083436` (+8,344 yüzde puan), macro-F1
  `+0,084008` (+8,401 yüzde puan).
- SmallCNN'e göre leaf accuracy `+0,040615` (+4,062 yüzde puan), macro-F1 `+0,040546`
  (+4,055 yüzde puan).

Eşleştirilmiş görüntü sonuçları:

| Karşılaştırma | İkisi doğru | Yalnız fine-tune doğru | Yalnız diğer model doğru | İkisi yanlış |
|---|---:|---:|---:|---:|
| Frozen-head ResNet18 | 17.771 | 1.660 | 33 | 36 |
| SmallCNN | 18.599 | 832 | 40 | 29 |

İki taraflı exact McNemar için frozen-head karşılaştırmasında `log10(p) = -439,86`, SmallCNN
karşılaştırmasında `p = 1,376 x 10^-193` elde edildi. Bunlar aynı sabit test yollarındaki farkın
çok büyük olduğunu gösteren post-hoc tanımlayıcı kanıtlardır; dış veri genellenebilirliği veya
klinik üstünlük kanıtı değildir.

## Eğitim eğrisi yorumu

- Validation macro-F1 epoch 1'de 0,9652 iken epoch 5'te 0,9939'a ve epoch 11'de 0,99595'e çıktı.
- Epoch 12 train macro-F1'ı biraz yükselirken validation macro-F1 0,99595'ten 0,99518'e düştü.
  Bu nedenle son epoch değil epoch 11 seçildi.
- Train ve validation metrikleri birbirine yakındır; eğride ağır bir overfitting çöküşü yoktur.
- Epoch 12'deki küçük validation düşüşü, checkpoint seçiminin validation ile yapılmasının neden
  gerekli olduğunu gösterir.

## Bağımsız artifact doğrulaması

- Review ZIP: `resnet18_lastblock_finetune_v1_review_package.zip`.
- Boyut: 6.532.295 byte.
- SHA-256: `702e1b019b64f8fb2e97a85fa409978ce6c70302bab5d17c8c0cb7fe7c74456f`.
- ZIP içindeki bütün 11 mevcut artifact'ın hash ve boyutları `artifact_hashes.json` ile eşleşti.
- Drive'da tutulan split ve iki checkpoint'in hash/boyut kayıtları manifestte mevcuttur.
- Training log doğru init checkpoint'i, 12 epoch'u, test aşamasını ve `RETURN CODE: 0` sonucunu
  doğruladı.
- Tahmin CSV'si 19.500 satır ve 19.500 benzersiz yol içerir.
- Frozen test manifestine eksik veya fazladan yol: 0.
- Her leaf desteği: tam 750.
- Probability anahtar, argmax, confidence ve toplam uyuşmazlığı: 0.
- Accuracy, macro-F1, balanced accuracy ve sınıf raporu tahminlerden yeniden hesaplanarak JSON ile
  eşleştirildi.
- Frozen-head ResNet18 ve SmallCNN review ZIP hash'leri yeniden doğrulandı; üç modelin test yolu
  kümeleri birebir aynıdır.

Kalıcı hafif kanıtlar `research/model_results/resnet18_lastblock_finetune_v1/` altındadır. Büyük
`best_model.pth`, `last_checkpoint.pth`, `test_predictions.csv`, kopya split ve review ZIP Drive'da
korunmalıdır.

## Bilimsel sınırlamalar

- Bu çok yüksek skor klinik kanser tanısı, kanser var/yok veya evre başarısı değildir.
- Parent sınıflar veri-seti/organ-modalite gruplarıdır; 67 leaf hatası parent seviyesinde gizlenir.
- Split bilinen exact ve reviewed aileleri birlikte tutar, fakat bilinmeyen bütün augmentation,
  patient veya scan-series ilişkilerini garanti etmez.
- Veri seti modality/source shortcut ve önceden augmentation riski taşır. Çok sayıda tam veya
  tama yakın sınıf sonucu bu bağlamda dikkatle yorumlanmalıdır.
- Tek seed ve tek veri seti vardır; dış veri seti veya bağımsız klinik merkez doğrulaması yoktur.
- Test seti önceki model karşılaştırmalarında görülmüştür. Fine-tuning sözleşmesi test sonucundan
  önce dondurulmuş ve checkpoint validation ile seçilmiş olsa da bu sonuç dokunulmamış yeni final
  test değil, shared-benchmark ablation'dır.

## Sonraki kontrollü adım

69 hata ve test-seti kalibrasyonu artık salt-okunur audit ile incelendi. Hataların 34'ü oral
normal/SCC arasındadır; 23 yanlış >=0,90, yedi yanlış >=0,99 confidence taşır. ECE `0,001032`,
top-2 accuracy `0,999949` ölçüldü; global ECE'nin kolay ve çok yüksek-confidence örneklerce
baskılandığı ayrıca kaydedildi. Ayrıntı: `research/RESNET18_LASTBLOCK_ERROR_ANALYSIS_V1_TR.md`.

Bir sonraki kontrollü adım, örnekleri önceden dondurulmuş bir Grad-CAM audit'i tasarlamaktır.
DenseNet121/EfficientNet-B0 veya GA deneyi ancak ayrı hipotez, bütçe ve onayla başlatılmalıdır.
