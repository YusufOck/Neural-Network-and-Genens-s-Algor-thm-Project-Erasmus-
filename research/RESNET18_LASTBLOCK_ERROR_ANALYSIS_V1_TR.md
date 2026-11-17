# ResNet18 Last-Block Hata ve Kalibrasyon Analizi v1

Tarih: 2026-11-17

Durum: Tamamlandı. Bu çalışma yeni eğitim değildir; doğrulanmış last-block ResNet18 koşusunun
19.500 test tahminini salt-okunur biçimde inceleyen post-hoc bir audit'tir.

## Amaç

Toplam accuracy tek başına hangi görüntülerin neden yanlış kaldığını göstermez. Bu audit şu
soruları cevaplamak için hazırlandı:

1. 69 leaf hatası hangi sınıf çiftlerinde yoğunlaşıyor?
2. Bu hataların kaçı sekiz parent veri-seti grubunu da yanlış yapıyor?
3. Model yanlışken ne kadar yüksek confidence verebiliyor?
4. Tahmin olasılıkları test setinde toplu olarak ne kadar kalibre görünüyor?
5. Hatalı görüntülerde gözle görülen ortak veri-seti özellikleri var mı?

## Sabit girdiler

- Review paketi: `resnet18_lastblock_finetune_v1_review_package.zip`.
- Paket SHA-256:
  `702e1b019b64f8fb2e97a85fa409978ce6c70302bab5d17c8c0cb7fe7c74456f`.
- Tahmin satırı: 19.500.
- Leaf sayısı: 26.
- Split SHA-256:
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Model checkpointi: validation macro-F1 ile seçilen epoch 11.

Script paket hash'ini, probability anahtarlarını, probability toplamlarını, argmax-prediction ve
confidence tutarlılığını kontrol eder. Dolu bir output klasörünün üzerine yazmayı reddeder.

## Ana hata sonucu

- Leaf hata: **69 / 19.500**.
- Parent hata: **2 / 19.500**.
- Leaf yanlışken parent doğru: **67 / 69**.
- Hatalar için oluşturulan temas sayfası: **12**.
- Tüm temas sayfaları ve 69 görüntü görsel QA sırasında açıldı.

En sık yönlü karışıklıklar:

| Gerçek leaf | Tahmin leaf | Sayı |
|---|---|---:|
| `oral_normal` | `oral_scc` | 18 |
| `oral_scc` | `oral_normal` | 16 |
| `breast_benign` | `breast_malignant` | 7 |
| `lymph_mcl` | `lymph_cll` | 5 |
| `brain_glioma` | `brain_menin` | 4 |
| `brain_menin` | `brain_tumor` | 4 |

`oral_normal` ile `oral_scc` arasındaki iki yönlü 34 hata, bütün 69 hatanın yaklaşık %49,3'ünü
oluşturur. Bu yüzden 0,996462 genel leaf accuracy çok yüksek olsa da kalan hata yükü sınıflar
arasında eşit dağılmamıştır.

## İki parent hatası

| Error ID | Gerçek | Tahmin | Confidence |
|---|---|---|---:|
| `ERR0001` | `lung_scc` | `breast_malignant` | 0,9860 |
| `ERR0002` | `breast_benign` | `oral_normal` | 0,7784 |

`ERR0001` modelin farklı bir parent gruba çok yüksek confidence ile geçtiği en ciddi örnektir.
`ERR0002` görüntüsünde büyük boş alan ve kenar boyunca belirgin germe/augmentasyon izi vardır.
Bu ikinci özellik temas sayfası üretiminden kaynaklanmaz; yerel kaynak JPEG'de de aynen görüldü.

## Yüksek-confidence hatalar

- Yanlış tahminlerin ortalama confidence değeri: **0,791409**.
- Confidence >= 0,90 olan yanlış: **23**.
- Confidence >= 0,99 olan yanlış: **7**.

Yedi >=0,99 hatanın leaf çiftleri şunlardır:

1. `oral_scc -> oral_normal`
2. `brain_glioma -> brain_menin` (iki görüntü)
3. `breast_benign -> breast_malignant` (iki görüntü)
4. `cervix_mep -> cervix_koc`
5. `breast_malignant -> breast_benign`

Bu bulgu, softmax confidence'ın "modelin kendi seçimine ne kadar yığıldığını" anlattığını; tıbbi
kesinlik veya yanlış yapmama garantisi olmadığını gösterir.

## Kalibrasyon sonuçları

| Ölçüt | Sonuç | Basit anlamı |
|---|---:|---|
| ECE, 15 eşit aralıklı bin | 0,001032 | Binlerde ortalama confidence ile gerçekleşen doğruluk farkının ağırlıklı ortalaması |
| Multiclass NLL | 0,010893 | Gerçek sınıfa düşük olasılık vermeyi logaritmik olarak cezalandırır |
| Multiclass Brier | 0,005630 | 26 sınıflı olasılık vektörü ile doğru one-hot hedef arasındaki karesel hata |
| Top-2 accuracy | 0,999949 | Gerçek leaf 19.499 / 19.500 görüntüde ilk iki tahmin içindedir |

Top-2 dışında kalan tek görüntü
`Breast Cancer/breast_malignant/breast_malignant_0298.jpg` dosyasıdır. Model burada
`breast_benign` tahminini 0,9922 confidence ile seçmiş; ikinci sıraya `cervix_dyk` koymuştur.
Gerçek `breast_malignant` sınıfı ilk ikiye girememiştir.

ECE'nin çok düşük olması dikkatle yorumlanmalıdır. Tahminlerin 19.275'i 0,9333-1,0 son bininde
toplanır ve bu büyük binin doğruluğu 0,99891, ortalama confidence'ı 0,99946'dır. Dolayısıyla global
ECE bu kalabalık ve kolay örnekler tarafından baskılanır. Daha az örnekli orta-confidence binlerinde
mutlak fark yaklaşık 0,02-0,096 arasında değişir. Sonuç "bu shared test üzerinde toplu kalibrasyon
iyi görünüyor" şeklinde söylenebilir; "her örnekte güvenilir" veya "klinik kalibre" denemez.

## 69 görüntünün görsel incelemesi

Görsel inceleme uzman patolog/radyolog değerlendirmesi değildir. Aşağıdaki gözlemler veri-seti ve
model davranışı hakkındaki mühendislik bulgularıdır:

- Hataların yaklaşık yarısı oral histoloji içindeki `normal`/`scc` ayrımındadır. Yanlışların iki
  yöne de gitmesi, yalnız tek sınıfa sistematik kayma olmadığını; bu iki leaf'in görsel sınırının
  veri setinde zor olduğunu düşündürür.
- Breast benign/malignant, lymphoma CLL/MCL ve brain glioma/meningioma/tumor hataları da çoğunlukla
  aynı parent ve aynı görüntü modalitesi içinde kalır.
- Oral ve lymphoma örneklerinin bir bölümünde, ayrıca `ERR0002` içinde görüntü kenarlarında
  uzatılmış piksel şeritleri görülür. Özgün dosyalar ayrıca açıldığında bu izlerin kaynak JPEG'lerde
  bulunduğu doğrulandı. Bunlar veri üretim/augmentasyon pipeline'ından kalmış olabilecek görsel
  artefaktlardır; temas sayfası renderer'ının hatası değildir.
- Tek bir görüntüye bakarak etiketin tıbben yanlış olduğunu söylemek mümkün değildir. Etiket
  gürültüsü şüphesi ancak kaynak veri/provenance veya uzman incelemesiyle doğrulanabilir.

## Üretilen artefaktlar

Klasör:
`research/model_results/resnet18_lastblock_finetune_v1/error_analysis_v1/`

- `summary.json`: ana hata ve kalibrasyon özeti.
- `error_rows.csv`: 69 hatanın izlenebilir tablosu ve tüm 26 probability değeri.
- `contact_sheet_index.csv`: Error ID ile sayfa eşlemesi.
- `contact_sheets/error_sheet_001.png` ... `error_sheet_012.png`: 69 görüntünün görsel sayfaları.
- `reliability_bins.csv`: 15 confidence bininin sayım/doğruluk/confidence değerleri.
- `reliability_diagram.png`: reliability grafiği.
- `hash_manifest.json`: 17 çıktı artefaktının SHA-256 ve boyut kaydı.

Manifestteki 17 dosyanın hash ve boyutları üretim sonrasında yeniden hesaplandı; uyuşmazlık 0'dır.

## Bilimsel sınırlar

- Bu shared benchmark test seti daha önce model karşılaştırmalarında kullanılmıştır; tamamen yeni
  dış doğrulama değildir.
- Kalibrasyon dış merkez, hasta düzeyi veya klinik kalibrasyon değildir.
- Split bilinen exact/review ailelerini ayırır; bilinmeyen augmentation, patient veya scan-series
  ilişkilerini tamamen dışladığını garanti etmez.
- Parent accuracy organ/modalite veri-seti grubu başarısıdır; kanser tanısı veya evre başarısı
  değildir.
- Görsel benzerlik, modelin gerçekten o bölgeyi kullandığını kanıtlamaz. Bunun için ayrı Grad-CAM
  deneyi gerekir.

## Sonraki kontrollü adım

Bu audit artık tamamlandığı için bir sonraki mantıklı adım, dondurulmuş fine-tuned ResNet18
checkpointi üzerinde önceden seçilmiş doğru/yanlış ve yüksek-confidence örnekler için Grad-CAM
deneyi tasarlamaktır. Grad-CAM yeni eğitim yapmaz; modelin tahmin sırasında hangi görüntü
bölgelerine duyarlı olduğunu yaklaşık olarak gösterir. Örnek seçimi ve kabul kuralları koddan önce
dondurulmalı, ardından Mehmet'in ayrı onayı alınmalıdır.
