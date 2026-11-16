# SmallCNN 26-Leaf Baseline Sonucu

Tarih: 2026-11-16

Durum: Tam kapsamlı Colab/T4 koşusu tamamlandı; indirilen review paketi, sabit split, 19.500 test
tahmini ve metrik aritmetiği yerelde bağımsız olarak doğrulandı.

Deney türü: Aynı sabit split üzerinde ResNet18 head-only baseline ile önceden sözleşmesi
dondurulmuş, sıfırdan eğitilen karşılaştırma. Klinik doğrulama değildir.

## Deney sözleşmesi

- Hedef: 26 leaf sınıfı; aynı tahminlerden sekiz parent veri-seti grubu türetildi.
- Model: Yaklaşık 97.050 eğitilebilir parametreli SmallCNN; pretrained ağırlık yoktur.
- Eğitim: Bütün model sıfırdan eğitildi.
- Girdi: 224 x 224, ImageNet normalizasyonu, yalnız train tarafında minimal augmentation.
- Optimizasyon: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Batch: 32; seed: 42.
- Azami epoch: 25; validation macro-F1 için patience: 5.
- Split: `practical_leakage_reduced_split_v1.csv`, SHA-256
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Split büyüklükleri: 91.000 train, 19.500 validation, 19.500 test; her leaf testte 750 örnek.
- Checkpoint seçimi: yalnız validation macro-F1. Test sonuçları model seçimi bittikten sonra
  üretildi.

## Ana sonuç

| Seviye | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| 26 leaf | 0,955846 | 0,955916 | 0,955846 |
| Türetilmiş 8 parent grup | 0,999231 | 0,998984 | 0,998800 |

- En iyi validation checkpoint'i: epoch 25.
- En iyi validation macro-F1: 0,955566.
- Test loss: 0,122079.
- Tamamlanan epoch: 25. Patience tetiklenmeden azami epoch'a ulaşıldı.
- Test macro-F1, en iyi validation macro-F1'den yalnız 0,000350 (0,035 yüzde puan) yüksektir.
- Leaf düzeyinde 18.639/19.500 doğru, 861 yanlış tahmin vardır.
- Parent düzeyinde 19.485/19.500 doğru, 15 yanlış tahmin vardır.
- Leaf yanlışlarının 846'sında parent yine doğrudur. Parent skorunun çok yüksek olmasının temel
  nedeni budur.

## En zayıf leaf sınıfları

| Sınıf | F1 |
|---|---:|
| `oral_scc` | 0,8451 |
| `oral_normal` | 0,8465 |
| `brain_menin` | 0,8595 |
| `brain_tumor` | 0,8935 |
| `kidney_normal` | 0,9286 |

Genel macro-F1 güçlü olsa da `oral_scc` ve `oral_normal` F1 değerleri yaklaşık 0,85 sınırının biraz
altındadır. Bu iki sınıf modelin başlıca zayıf noktasıdır; yalnız toplam accuracy ile gizlenmemelidir.

## En önemli karışmalar

1. `oral_normal -> oral_scc`: 127.
2. `kidney_normal -> kidney_tumor`: 100.
3. `oral_scc -> oral_normal`: 98.
4. `brain_tumor -> brain_menin`: 80.
5. `brain_menin -> brain_tumor`: 68.
6. `brain_glioma -> brain_menin`: 67.
7. `lung_aca -> lung_scc`: 62.
8. `cervix_koc -> cervix_dyk`: 43.
9. `lung_scc -> lung_aca`: 33.
10. `cervix_mep -> cervix_dyk`: 27.

Yanlış tahminlerin ortalama güveni 0,6897, doğru tahminlerin ortalama güveni 0,9484'tür. Bununla
birlikte 861 yanlışın 79'u en az 0,90 güvenle yapılmıştır; yüksek confidence tek başına doğruluk
kanıtı değildir.

## Aynı test setinde ResNet18 karşılaştırması

Her iki modelin de aynı 19.500 test yolunu ve aynı gerçek etiketleri kullandığı doğrulandı.

| Ölçüt | Frozen-head ResNet18 | SmallCNN | SmallCNN farkı |
|---|---:|---:|---:|
| Leaf accuracy | 0,913026 | 0,955846 | +0,042821 |
| Leaf macro-F1 | 0,912455 | 0,955916 | +0,043461 |
| Parent accuracy | 0,997744 | 0,999231 | +0,001487 |

Eşleştirilmiş test görüntülerinde iki model de 17.288 görüntüyü doğru, 345 görüntüyü yanlış bildi.
Yalnız SmallCNN'in doğru bildiği 1.351; yalnız ResNet18'in doğru bildiği 516 görüntü vardır.
SmallCNN'in net doğru tahmin kazancı 835 görüntüdür.

İki taraflı exact McNemar testi için `p ≈ 6,09 x 10^-86` bulunmuştur. Bu, aynı test görüntülerinde
SmallCNN avantajının rastlantısal eşleşme farkıyla açıklanmasının çok güç olduğunu gösteren post-hoc
tanımlayıcı kanıttır. Tek bir veri seti/split üzerindeki bu test klinik genellenebilirlik veya bütün
CNN'ler için evrensel üstünlük kanıtı değildir.

SmallCNN özellikle ResNet18'in zayıf olduğu `lymph_mcl`, `lymph_fl` ve `lymph_cll` sınıflarında
büyük F1 artışı sağlamıştır. Buna karşılık `brain_tumor`, `kidney_normal`, `brain_menin`,
`kidney_tumor` ve `brain_glioma` sınıflarında F1 azalmıştır.

## Sonucun doğru yorumu

SmallCNN bu deneyde ResNet18 head-only baseline'dan daha başarılıdır. Bunun olası teknik nedeni,
SmallCNN'in yaklaşık 97 bin parametresinin tamamının bu veri setine uyarlanması; ResNet18 koşusunda
ise ImageNet omurgasının dondurulup yalnız yaklaşık 13 bin parametreli son başlığın öğrenmesidir.
Bu bulgu "küçük ağ her zaman ResNet'ten iyidir" anlamına gelmez. Veri setindeki doku, modalite,
kaynak ve ön işleme ipuçlarına küçük modelin kolay uyum sağlaması da sonuca katkıda bulunabilir.

Parent accuracy `0,999231` kanser var/yok, kanser evresi veya klinik tanı doğruluğu değildir. Sekiz
parent sınıf veri-seti/organ-modalite gruplarıdır. Örneğin `oral_normal` görüntüsünün `oral_scc`
tahmin edilmesi leaf düzeyinde yanlışken, ikisi de Oral Cancer parent'ında olduğu için parent
düzeyinde doğru sayılır.

## Eğitim eğrisi yorumu

- Validation macro-F1 epoch 25'te en yüksek değerine ulaştı; bu nedenle seçilen checkpoint 25'tir.
- Validation skorunun train skorundan yüksek olması tek başına hata değildir. Train metriği
  augmentation ve dropout açıkken daha zor örneklerde; validation ise temiz görüntülerde ve eval
  modunda ölçülür.
- En iyi değerin azami epoch'ta çıkması, modelin daha uzun eğitimle ilerleyebileceğini düşündürebilir;
  fakat test sonucunu gördükten sonra bu sözleşmeyi uzatmak doğru değildir. Böyle bir deney yeni ve
  önceden dondurulmuş bir sözleşme olmalıdır.

## Bağımsız artifact doğrulaması

- Review ZIP boyutu: 9.148.000 byte.
- Review ZIP SHA-256: `9d5294332a25b9fd4909c6f1c9964f795ff549abd639b85ac0a32a9a2500eb33`.
- Ayrı indirilen ve ZIP içindeki `artifact_hashes.json` birebir aynıdır; ayrı dosyanın SHA-256 değeri
  `da3b6ce1644c353bf6bf986cde7ba928249d28979e644607d30e53739ad144da`.
- Manifestte listelenen dokuz artifact'ın hash ve boyutları eşleşti.
- Eğitim log'u epoch 14 sonrası kesintiyi, epoch 15'ten doğru resume'u ve epoch 25 sonrası tamamlanan
  test aşamasını gösterir.
- Split tam 91.000/19.500/19.500'dür; 130.000 yol benzersizdir, her leaf 3.500/750/750'dir ve
  cross-split family sayısı 0'dır.
- Tahmin dosyası 19.500 benzersiz test yolundan oluşur; test dışı yol, label, olasılık, argmax,
  prediction ve confidence uyuşmazlığı 0'dır.
- Accuracy, macro-F1, balanced accuracy ve sınıf raporu tahminlerden yeniden hesaplanarak doğrulandı.

Kalıcı hafif kanıtlar `research/model_results/smallcnn_26leaf_baseline_v1/` altındadır. Büyük
`best_model.pth`, `last_checkpoint.pth`, `test_predictions.csv` ve kopya split manifest Drive'da
tutulmalıdır; checkpoint klasörü silinmemelidir.

## Sınırlamalar ve sonraki kontrollü adım

- Split bütün bilinen exact/review ailelerini birlikte tutar; bilinmeyen bütün near-duplicate,
  patient veya series ilişkilerini garanti etmez.
- Veri seti kaynak/modalite confounding taşır.
- Sonuç yalnız bu veri seti ve sabit split için eğitim amaçlı performanstır; klinik kullanım kanıtı
  değildir.
- Her model için tek seed vardır; seed varyansı ölçülmemiştir.
- Hesap bütçeleri birebir eşit değildir: SmallCNN en fazla 25, ResNet18 en fazla 15 epoch aldı.
- Karşılaştırılan ResNet18 yalnız head-only modeldir; kontrollü last-block fine-tuning sonucu değildir.
- SmallCNN en iyi skorunu son epoch'ta verdiği için eğitim süresi sınırına duyarlılık ayrıca
  incelenebilir, ancak test setine bakılarak mevcut deney değiştirilemez.

Sıradaki deney otomatik başlamamalıdır. En anlamlı aday, aynı splitte validation ile seçilen,
önceden sözleşmesi dondurulmuş ResNet18 last-block fine-tuning koşusudur. Bu deney, head-only
ResNet18'in veri setine yeterince uyum sağlayamaması hipotezini test eder. Test seti hiperparametre
seçimi için kullanılmamalıdır.
