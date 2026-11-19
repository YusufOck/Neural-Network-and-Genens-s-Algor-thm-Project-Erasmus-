# GA ve Random Kazananları Tam-Veri Validation-Only Doğrulama Planı

Tarih: 2026-11-19

Durum: Sözleşme, tam train/validation manifesti, restart-safe runner ve kilitli Colab notebook'u
donduruldu. Gerçek 130.000 dosyalık envanter doğrulaması ve 237/237 proje testi geçti. Tam eğitim
başlatılmadı ve test görüntüsü açılmadı.

Deney türü: `full_data_validation_only_hyperparameter_confirmation`.

## 1. Araştırma sorusu

Dört epoch ve 9.100/3.900 proxy üzerinde GA kazananı random-search kazananından yalnız yaklaşık
0,267 yüzde puan yüksek validation macro-F1 üretmiştir. Şimdi şu soru test edilecektir:

> Proxy'de seçilen GA ve random-search ayarları, aynı başlangıç checkpointinden ve aynı tam
> 91.000 train / 19.500 validation verisinden eğitildiğinde hangi ayar daha yüksek validation
> macro-F1 üretir; ikisi önceki elle seçilmiş last-block ayarıyla nasıl karşılaştırılır?

Bu aşama test performansı ölçmez. Amaç, küçük proxy sıralamasının tam train/validation verisinde
korunup korunmadığını validation-only olarak incelemektir.

## 2. Dondurulmuş adaylar

| Yöntem | Candidate | Learning rate | Weight decay | Dropout | Proxy validation macro-F1 |
|---|---|---:|---:|---:|---:|
| GA | `GA_G01_C02` | `7,24655842233e-05` | `1,94540732384e-05` | `0,461905683802` | 0,952281 |
| Random search | `RANDOM_002` | `5,86477956786e-05` | `3,14775256001e-04` | `0,282717237583` | 0,949612 |

Bu değerler tamamlanmış search paketindeki `search_summary.json` dosyasından alınmış ve yerelde
config ile çapraz doğrulanmıştır. Kaynak summary SHA-256:

`debb373153f0c3264e599e6c91a00c1689a86d2a21730cee2bba6d70f4a36f9c`

Kaynak review ZIP SHA-256:

`817c95918dfc997ab75ef8410810d2b86afe44241d72cc4138b160443d804f1b`

## 3. İki adayda aynı kalacak eğitim sözleşmesi

- Model: ResNet18.
- Hedef: 26 leaf sınıf.
- Başlangıç: tamamlanmış head-only ResNet18 epoch-9 best checkpoint.
- Init SHA-256:
  `20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177`.
- Eğitilebilir bölüm: yalnız `layer4 + fc`.
- Eğitilebilir parametre: 8.407.066.
- Görüntü: 224 x 224 ve ImageNet normalization.
- Augmentation: minimal horizontal flip + ±5 derece rotation.
- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau.
- Batch: 32.
- Seed: 42.
- Azami epoch: 12.
- Early-stopping patience: 4.
- Checkpoint seçimi: yalnız en yüksek validation macro-F1.

Yalnız üç aranan hiperparametre adaylar arasında farklıdır: learning rate, weight decay ve
dropout. Bu sayede karşılaştırma hesaplama ve veri bakımından eşittir.

## 4. Tam train/validation manifesti

Kaynak practical split değişmemiştir:

`research/splits/practical_leakage_reduced_split_v1.csv`

Kaynak SHA-256:

`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`

Yeni validation-only manifest:

`research/splits/ga_full_confirmation_v1.csv`

SHA-256:

`afe9d336d53ae24490e065c37b5cc764d0cfeff8f01b3fb7ee8af756bcd375f9`

Builder bütün frozen train ve validation satırlarını kopyalar, test satırlarının tamamını atar ve
hiçbir görüntü açmaz:

- Train: 91.000; her leaf 3.500.
- Validation: 19.500; her leaf 750.
- Test: 0.
- Leaf: 26.
- Benzersiz yol: 110.500/110.500.

Manifest gerçek yerel 130.000 dosya envanterine karşı `load_fixed_split_manifest` ve
`load_search_proxy_manifest` ile doğrulanmıştır.

## 5. Test güvenliği

Tam confirmation, mevcut hash-korumalı validation-only trainer motorunu kullanır. Yeni manifestte
test satırı bulunmadığı için:

- test `RecordDataset` oluşturulmaz;
- test DataLoader oluşturulmaz;
- test görüntüsü açılmaz;
- `test_metrics.json` üretilemez;
- test fitness veya seçimde kullanılamaz;
- bütün candidate özetlerinde `test_evaluated=false` zorunludur.

Trainer'ın iç artifact adı geriye dönük uyumluluk nedeniyle `search_trial_summary.json` olarak
kalır. Üst runner bu çıktıyı `full_data_validation_only_hyperparameter_confirmation` sözleşmesi
altında yeniden doğrular. Bu isim ayrıntısı test kullanılmasını sağlamaz veya deney anlamını
değiştirmez.

## 6. Kesinti ve devam

- GA ve random adayları Drive'da ayrı klasörlere yazılır.
- Candidate sözleşmesi eğitim başlamadan `candidate.json` içine yazılır.
- Tamamlanmış aday yeniden çalıştırılmaz.
- Yarım aday `last_checkpoint.pth` üzerinden optimizer, scheduler, history ve RNG durumuyla devam
  eder.
- Yarım artifact olup checkpoint yoksa otomatik overwrite yapılmaz; runner durur.
- Büyük `best_model.pth` ve `last_checkpoint.pth` dosyaları Drive'da korunur, hafif review ZIP'ine
  eklenmez.

## 7. Çıktılar ve karar

Üst düzey kanıtlar:

- `run_contract.json`
- `frozen_config.json`
- `confirmation_results.csv`
- `confirmation_summary.json`
- Her aday için candidate, result, validation-only summary, history, run config ve console log.

İki aday tam validation macro-F1 ile karşılaştırılır. Önceden tamamlanmış elle seçilmiş last-block
baseline'ın validation macro-F1 değeri `0,9959480748813776` yalnız tanımlayıcı referans olarak
raporlanır. Test sonucuna bakarak aday veya checkpoint seçilmez.

Tam validation kanıtları birlikte incelenmeden test değerlendirmesi hazırlanmayacaktır. Daha sonra
ortak test setinin açılması ayrı hipotez, ayrı kod ve ayrı Mehmet onayı gerektirir.

## 8. Colab akışı ve maliyet

Notebook:

`notebooks/ga_random_full_confirmation_colab.ipynb`

İlk üç kod hücresi yalnız Drive/repo/veri/hash/GPU/sözleşme preflight yapar. Uzun hücre varsayılan
olarak:

`AUTHORIZE_FULL_CONFIRMATION = False`

durumundadır. İki tam eğitim T4 üzerinde toplam birkaç saat sürebilir; kesin süre Colab I/O ve
erken durdurmaya bağlıdır. Preflight incelemesinden ve ayrı çalıştırma onayından önce kilit
açılmayacaktır.

## 9. Dondurulmuş uygulama hash'leri

- Trainer:
  `807a23e40ea3aaa5c0c98737beddb637d71c5e5a9e5f230aa0b75d0be7212a77`
- Manifest builder:
  `68664894ab314bbf0fee870e25b15d6c7392a1b21fdc2635c01254e07582d0cb`
- Confirmation runner:
  `69d045c449245b1e408b33017737691c153c25586461d3cf5c8691c2a449fc49`
- Frozen config:
  `7f0886527f24365373842081561a0a4b6318f16fb59d0272fff1266a9bbd34a9`
- Confirmation manifest:
  `afe9d336d53ae24490e065c37b5cc764d0cfeff8f01b3fb7ee8af756bcd375f9`
- Manifest report:
  `33ee90d0c62320b02a3c2e9e0e0db3c5662c759d6b1ddb8cef9dcb801d904146`
- Colab notebook:
  `4f488a3d0dca16c5fb9e4274f0e7d90ecf509f616256a14cb460d655fb024233`

## 10. Doğrulama durumu

- Yeni odak testleri: 11/11 geçti.
- Tam proje test paketi: 237/237 geçti.
- Notebook JSON'u temiz; bütün kod hücreleri derleniyor ve çıktı/execution count taşımıyor.
- Gerçek envanter: 130.000 dataset yolu; 110.500 benzersiz confirmation yolu.
- Kaynak search winner eşleşmesi: GA ve random 2/2.
- Tam eğitim: başlamadı.
- Test değerlendirmesi: yapılmadı.
