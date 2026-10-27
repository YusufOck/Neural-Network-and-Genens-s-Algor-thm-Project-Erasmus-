# Yakın-kopya aday üretimi ölçeklenebilirlik pilotu v2

Tarih: 2026-10-27

Durum: **Kalibrasyon pilot kabul kapısı geçti. Robust ResNet18 top-10, 1.094/1.094 aynı-kaynak
referansını bulurken 224.926 olası çiftin 18.012'sini (%8,01) aday yaptı. Bu sonuç tam veri
taraması, augmentation-family üretimi veya classifier split'i değildir.**

## 1. Bu adımın amacı

Frozen duplicate rule v1 bir çift kendisine verildiğinde güçlü çalışıyor; fakat 130.000 görüntünün
bütün çiftlerini bu pahalı doğrulayıcıyla karşılaştırmak mümkün değildir. Candidate generator'ın
görevi, aynı-kaynak olabilecek çiftleri kaçırmadan küçük bir kısa liste oluşturmaktır:

```text
görüntüler
  ↓ ucuz candidate generator
yüksek-recall kısa liste
  ↓ frozen ORB/RANSAC/ECC duplicate rule v1
same_source / independent / uncertain
```

V1'de pHash tam recall'a ancak çiftlerin %94,50'sini geçirerek ulaşmış; standart ResNet18 top-50
ise daha az aday üretmesine rağmen beş kontrollü akrabayı kaçırmıştı. V2 özellikle bu karanlık ve
geometrik olarak kaydırılmış örnekleri hedefledi.

## 2. Girdi ve deney türü

Bu çalışma bir **candidate-retrieval kalibrasyon pilotudur**; model eğitimi değildir.

- 1.342 etiketli kalibrasyon çifti;
- 1.748 benzersiz görüntü;
- 1.094 aynı-kaynak referansı;
- 247 bağımsız referans;
- 1 belirsiz referans;
- uyumlu `source_group` sınırları içindeki 224.926 olası çift.

Aynı-kaynak referanslarının 1.064'ü kontrollü sentetik/exact pozitif, 30'u AI-destekli görsel
incelemeyle kabul edilmiş doğal çifttir. Önceki duplicate-rule holdout'u artık açılmış olduğu için
bu havuz yeni ve dokunulmamış bir test kümesi değildir.

Komut:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\benchmark_candidate_generation_v2.py `
  --batch-size 8 `
  --workers 4 `
  --seed 42
```

RTX 3050 Laptop GPU üzerindeki doğrulanmış koşu 138,75 saniye sürdü. ResNet18 ImageNet checkpoint
SHA-256 değeri
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` olarak kaydedildi.

## 3. V2 neyi değiştirdi?

ImageNet'te önceden eğitilmiş ResNet18'in son sınıflandırma katmanı kaldırıldı. Ağ burada kanser
sınıfı tahmin etmiyor; her görüntüyü 512 sayılık normalize bir görsel özete dönüştürüyor.

Her görüntü için dört ayrı descriptor üretildi:

1. ham tam görüntü + yatay çevirme ortalaması;
2. `%1` cutoff ile autocontrast uygulanmış tam görüntü + yatay çevirme ortalaması;
3. ham görüntünün merkez/sol/sağ/üst/alt `%80` crop'larının ve yatay çevrilmiş hallerinin ortalaması;
4. aynı multi-crop işleminin autocontrast uygulanmış hali.

`Autocontrast`, aşırı karartılmış görüntüde kalan düşük dinamik aralığı yeniden yayar. `Multi-crop`
ise aynı yapının kaydırılmış veya kenarı kırpılmış halinde ortak iç bölgelere daha dayanıklı bir
özet üretir. Her descriptor kanalı kendi içinde komşu arar; kanallardan herhangi biri çifti top-k
içine alırsa çift tek bir benzersiz aday olarak tutulur.

Ek olarak, robust top-k adayları ile flip-aware pHash yarıçaplarının birleşimleri de ölçüldü.
Birleşim (`union`) nihai duplicate kararı değildir; iki kanaldan en az biri bulduğunda çifti kısa
listeye ekler.

## 4. Koşudan önce belirlenen pilot kabul kapısı

Bir yapılandırmanın pilot kapısını geçmesi için:

- 1.094 aynı-kaynak referansında `%100` candidate recall;
- 30 doğal aynı-kaynak çiftinde `%100` candidate recall;
- 224.926 olası çiftin en fazla `%25`'ini aday yapması

gerekiyordu. Bu ölçüt yalnız kalibrasyon pilotunu değerlendirir; tam veri setine otomatik geçiş
izni değildir.

## 5. Robust ResNet18 sonucu

| Yapılandırma | Benzersiz aday çift | Olası çiftlerin oranı | Aynı-kaynak hit | Recall | Doğal hit | Bağımsızların aday oranı |
|---|---:|---:|---:|---:|---:|---:|
| robust top-1 | 3.222 | %1,43 | 351/1.094 | %32,08 | 30/30 | %8,50 |
| robust top-5 | 11.061 | %4,92 | 1.082/1.094 | %98,90 | 30/30 | %24,29 |
| **robust top-10** | **18.012** | **%8,01** | **1.094/1.094** | **%100** | **30/30** | **%32,39** |
| robust top-20 | 38.152 | %16,96 | 1.094/1.094 | %100 | 30/30 | %46,96 |
| robust top-50 | 83.003 | %36,90 | 1.094/1.094 | %100 | 30/30 | %70,04 |

En küçük kabul edilen yapılandırma `robust top-10` oldu. Top-5 yalnız 12 kontrollü örneği kaçırdı:
10 `combined_stress` ve 2 `brightness_1_20`. Top-10'da bütün kontrollü dönüşüm türleri 104/104,
24 SHA-256 exact pozitif 24/24 ve doğal çiftler 30/30 bulundu.

Bağımsız referansların 80/247'sinin aday listesine girmesi false duplicate kararı değildir. Aday
üretici bilerek recall ağırlıklıdır; bu çiftleri frozen duplicate rule v1'in reddetmesi beklenir.

## 6. V1'de kaçan beş çift ne oldu?

| Pair ID | Dönüşüm | V1 ResNet union rank | V2 en iyi rank | V2'de etkili descriptor |
|---|---|---:|---:|---|
| C000053 | `brightness_0_20` | 98 | 1 | autocontrast multi-crop |
| C000453 | `brightness_0_20` | 55 | 3 | autocontrast full |
| C000684 | `brightness_0_20` | 64 | 2 | autocontrast full |
| C001146 | `combined_stress` | 100 | 6 | autocontrast multi-crop |
| C001151 | `shift_plus10_minus10` | 71 | 2 | raw multi-crop |

Bu tablo v2 değişikliklerinin amaçlanan arıza türlerini doğrudan düzelttiğini gösterir. Özellikle
karanlık örneklerde autocontrast, kaydırılmış örnekte multi-crop etkili oldu.

## 7. pHash birleşimi gerekli oldu mu?

`robust top-10 + pHash <= 8` 18.053 aday üretti ve yine 1.094/1.094 recall verdi. pHash kanalı
kalibrasyonda ek bir pozitif bulmadığı için en küçük kabul edilen yöntem robust top-10 olarak kaldı.
Daha geniş pHash yarıçapları aday yükünü artırdı; örneğin `top-10 + pHash <= 24` 59.294 adayla
%26,36'ya çıktığı için önceden belirlenen yük sınırını geçti.

Bu, pHash'in hiçbir değeri olmadığı anlamına gelmez. SHA-256 exact kontrolü ve düşük pHash
mesafeleri tam tarama tasarımında ayrı güvenlik/öncelik kanalı olabilir; fakat v2 kalibrasyon
recall'ını artırmadılar.

## 8. Bütünlük ve test doğrulaması

- Projenin 47/47 Python 3.10 birim testi geçti.
- `pair_candidate_metrics.csv`: 1.342 satır, 1.342 benzersiz Pair ID.
- `image_index.csv`: 1.748 satır.
- Dört descriptor matrisi de `(1748, 512)` `float32`; NaN/Inf yok.
- Bütün descriptor normları yaklaşık 1,0.
- Script, çift tablosu, development kararları ve holdout kararlarının kaydedilmiş SHA-256
  değerleri mevcut dosyalarla eşleşti.
- Rank sütunu için eşit cosine skorlarında candidate generator ile aynı stable-order davranışını
  zorlayan ayrı regresyon testi eklendi.

Yerel çıktılar:

- `artifacts/audit/candidate_generation_benchmark_v2/report.json`
- `artifacts/audit/candidate_generation_benchmark_v2/pair_candidate_metrics.csv`
- `artifacts/audit/candidate_generation_benchmark_v2/image_index.csv`
- `artifacts/audit/candidate_generation_benchmark_v2/resnet18_robust_descriptors.npz`

## 9. Bilimsel karar ve sınırlar

Candidate generator v2, mevcut kalibrasyon havuzundaki kabul kapısını geçti. Bu kanıt şunları
söylemeye yeter:

- brightness-normalized ve multi-crop descriptor'lar v1'in bilinen beş hatasını düzeltti;
- robust top-10 bu havuzda tam recall ile pHash-32'ye göre çok daha küçük aday listesi üretti;
- v2 daha büyük ölçekli retrieval/capacity denemesine geçmeye değer.

Şunları söylemeye yetmez:

- 130.000 görüntüde recall'ın `%100` kalacağı;
- bütün augmentation-family'lerin bulunduğu;
- frozen duplicate rule'ın milyonlarca adayı makul sürede skorlayacağı;
- leakage-aware split'in hazır olduğu;
- kanser sınıflandırma modelinin eğitildiği veya doğruluğunun ölçüldüğü.

Kalibrasyon grupları gerçek source-group'lardan çok daha küçüktür. Tam veri havuzunda komşu
yoğunluğu artınca gerçek akrabanın rank'i kötüleşebilir. Ayrıca dört descriptor kanalında top-10
arama, 130.000 görüntü için en fazla yaklaşık 5,2 milyon yönlü komşuluk kaydı üretebilir; benzersiz
çift sayısı daha düşük olsa da pahalı frozen doğrulayıcıya doğrudan gönderilmesi hâlâ günler
sürebilir.

## 10. Sonraki kontrollü adım

Tam taramadan önce `scaled retrieval/capacity pilot` yapılmalıdır:

1. dört descriptor kanalını kaynak grup bazında bloklu veya approximate-nearest-neighbor indeksle;
2. bütün 26 yaprağı temsil eden daha büyük, fakat sınırlı bir görüntü havuzunda robust top-10
   candidate yoğunluğunu ve bilinen akraba rank'lerini ölç;
3. frozen rule v1'e gitmeden önce ucuz bir ikinci-aşama eleme/önceliklendirme olup olamayacağını
   ölç;
4. runtime, disk, GPU/CPU ve tahmini tam-tarama maliyetini kaydet;
5. kabul koşulu sağlanmadan 130.000 görüntü taramasını ve family/split üretimini başlatma.

Bu v2 sonucu bir duplicate-audit altyapı sonucudur; kanser sınıflandırma performansı değildir.
