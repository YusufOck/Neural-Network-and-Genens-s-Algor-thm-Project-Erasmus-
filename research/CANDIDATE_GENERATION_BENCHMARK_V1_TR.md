# Yakın-kopya aday üretimi ölçeklenebilirlik pilotu v1

Tarih: 2026-10-26

Durum: **Pilot tamamlandı; mevcut pHash ve ResNet18 top-k yöntemlerinin hiçbiri tek başına
130.000 görüntülük tarama için kabul edilmedi. Tam veri taraması, family ve split başlatılmadı.**

Bu pilotun amacı nihai duplicate kararı vermek değildir. Amaç, pahalı ORB/RANSAC/ECC doğrulayıcıya
gönderilecek küçük ve yüksek-recall bir aday listesi üretmenin mümkün olup olmadığını ölçmektir.

## 1. Neden aday üretici gerekiyor?

130.000 görüntünün bütün çiftlerini karşılaştırmak yaklaşık 8,45 milyar çift demektir. Bunun yerine
iki aşamalı sistem gerekir:

```text
130.000 görüntü
    ↓ ucuz ve yüksek-recall aday bulucu
azaltılmış çift listesi
    ↓ frozen ORB/RANSAC/ECC duplicate rule v1
same_source / independent / uncertain
    ↓
parçalanmayan görüntü aileleri
```

Aday bulucu yalnız kısa liste hazırlar. Bir çifte aday demek, “aynı kaynak” kararı vermek değildir.

## 2. Pilot girdisi ve komut

Mevcut 1.342 kalibrasyon çifti 1.748 benzersiz görüntü içerir:

- 1.094 aynı-kaynak referansı;
- 247 bağımsız görsel referansı;
- 1 belirsiz referans.

Aynı-kaynakların 1.064'ü kontrollü sentetik/exact pozitif, 30'u AI-assisted doğal görsel adaydır.

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\benchmark_candidate_generation.py `
  --batch-size 32 `
  --workers 4 `
  --seed 42
```

Çalışma NVIDIA RTX 3050 Laptop GPU üzerinde 28,68 saniye sürdü. ResNet18 ImageNet checkpoint
SHA-256 değeri:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

Yerel çıktılar:

- `artifacts/audit/candidate_generation_benchmark_v1/report.json`
- `artifacts/audit/candidate_generation_benchmark_v1/pair_candidate_metrics.csv`
- `artifacts/audit/candidate_generation_benchmark_v1/image_index.csv`
- `artifacts/audit/candidate_generation_benchmark_v1/resnet18_embeddings.npz`

## 3. Karşılaştırılan yöntemler

### Flip-aware pHash

Her görüntünün orijinal ve yatay çevrilmiş 64-bit pHash'i çıkarıldı. İki görüntü arasındaki dört
yönelim karşılaştırmasının en küçük Hamming mesafesi kullanıldı. Yarıçaplar 8, 12, 16, 24 ve 32
olarak denendi.

### ResNet18 top-k retrieval

ImageNet'te önceden eğitilmiş ResNet18'in son sınıflandırma katmanı kaldırıldı. Her görüntü 512
sayıdan oluşan normalize bir özellik vektörüne dönüştürüldü. Orijinal/yatay-flip benzerliklerinin
en yükseği kullanıldı. Her görüntü için en yakın 1, 5, 10, 20 ve 50 komşu aday yapıldı.

Bu kullanım model eğitimi değildir. ResNet18 burada yalnız benzer görüntü arama özelliği çıkarır.

## 4. pHash sonucu

Kalibrasyon havuzunda uyumlu source-group içindeki toplam olası çift sayısı 224.926'dır.

| Yöntem | Aday çift | Aynı-kaynak recall | Doğal pozitif recall | Bağımsız çift aday oranı |
|---|---:|---:|---:|---:|
| pHash ≤8 | 1.491 | %46,7 | %26,7 | %2,8 |
| pHash ≤12 | 2.541 | %59,5 | %53,3 | %6,1 |
| pHash ≤16 | 4.553 | %71,6 | %93,3 | %15,0 |
| pHash ≤24 | 49.032 | %87,8 | %96,7 | %45,7 |
| pHash ≤32 | 212.562 | %100 | %100 | %98,0 |

pHash ≤32 hiçbir pozitif kaçırmadı; fakat 224.926 olası çiftin 212.562'sini (%94,50) aday yaptı.
Bu, hesap yükünü yeterince azaltmadığı için ölçeklenebilir aday üretici sayılmaz. Önceki pHash ≤12
kuralı ise pozitiflerin yaklaşık %40'ını kaçırdığı için recall bakımından kabul edilemez.

## 5. ResNet18 sonucu

| Yöntem | Aday çift | Aynı-kaynak hit/toplam | Recall | Doğal pozitif recall | Bağımsız çift aday oranı |
|---|---:|---:|---:|---:|---:|
| top-1 | 1.459 | 263/1.094 | %24,04 | %96,67 | %5,26 |
| top-5 | 6.436 | 971/1.094 | %88,76 | %100 | %16,19 |
| top-10 | 11.422 | 1.048/1.094 | %95,80 | %100 | %23,48 |
| top-20 | 23.478 | 1.080/1.094 | %98,72 | %100 | %33,60 |
| top-50 | 58.000 | 1.089/1.094 | %99,54 | %100 | %54,25 |

Top-50 bile beş kontrollü pozitif akrabayı kaçırdı:

- üç `brightness_0_20` örneği;
- bir `combined_stress` örneği;
- bir `shift_plus10_minus10` örneği.

Bu sonuç ImageNet özelliklerinin doğal aynı-kaynak örneklerini iyi sıraladığını, fakat aşırı
karartma ve geometrik kayma gibi kontrollü zor dönüşümlerde tek başına yeterli olmadığını gösterir.

## 6. Neden tam tarama başlatılmadı?

Top-k yönteminin 130.000 görüntü için yönlü üst sınırı:

- top-5: 650.000;
- top-20: 2.600.000;
- top-50: 6.500.000 aday ilişki.

Gerçek benzersiz çift sayısı karşılıklı komşuluk nedeniyle bu üst sınırdan düşük olabilir. Buna
rağmen frozen doğrulayıcının ölçülmüş yaklaşık 0,815 saniye/çift CPU süresiyle adayları doğrudan
skorlamak günler veya haftalar alabilir. Ayrıca top-5 ve top-20 pozitif kaçırdığı için yalnız hız
uğruna seçilemez.

Kalibrasyon havuzu tam veri setinden çok daha küçüktür. 130.000 görüntüde komşu yoğunluğu artacağı
için aynı akrabanın rank'i daha da kötüleşebilir. Bu nedenle buradaki top-k aday sayıları tam veri
sonucu olarak extrapolate edilmemelidir.

## 7. Bilimsel karar

Candidate generator v1 **reddedilmedi**, fakat tek başına yeterli bulunmadı:

- pHash düşük eşikte hızlı fakat recall yetersiz;
- pHash yüksek eşikte recall tam fakat neredeyse bütün çiftleri geçiriyor;
- ResNet18 retrieval doğal adaylarda güçlü fakat aşırı parlaklık/geometri dönüşümlerini kaçırıyor;
- cosine similarity dağılımları aynı-kaynak ve bağımsız çiftlerde örtüşüyor; tek similarity eşiği
  frozen duplicate kararının yerine kullanılamaz.

Bu pilot, tam taramaya geçmeden önce ikinci bir aday kanalı gerektiğini kanıtladı.

## 8. Sonraki kontrollü adım

Candidate generator v2 için şu değişiklikler development/calibration pilotunda test edilecek:

1. parlaklık/kontrast normalize edilmiş ResNet18 embedding varyantı;
2. kayma/kırpmaya dayanıklılık için merkez ve hafif kaydırılmış crop embeddingleri;
3. ResNet top-k ile düşük-yük pHash adaylarının birleşimi;
4. her yöntemde pozitif recall, aday sayısı ve dönüşüm-bazlı kaçırma tablosu;
5. kabul koşulu belirlenmeden 130.000 görüntülük taramayı başlatmama.

Bu bir pilot sonucudur; augmentation family veya classifier split değildir.
