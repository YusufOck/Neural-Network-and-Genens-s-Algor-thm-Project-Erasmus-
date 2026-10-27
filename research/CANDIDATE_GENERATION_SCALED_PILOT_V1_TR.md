# Candidate Generator v2 ölçekli retrieval/capacity pilotu

Tarih: 2026-10-27

Durum: **Önceden belirlenen robust top-10 kabul kapısı geçmedi. Maksimum 8.248 görüntülük
indekste 1.092/1.094 aynı-kaynak referansı bulundu; iki `brightness_1_20` kontrollü çift
top-10 ve top-20 dışında kaldı. Tam 130.000 görüntü taraması başlatılmadı.**

## 1. Bu pilot neden yapıldı?

Candidate Generator v2, 1.748 kalibrasyon görüntüsünde robust top-10 ile 1.094/1.094 recall
sağlamıştı. Fakat daha büyük bir indeks daha fazla benzer distractor içerir. Küçük havuzda ilk
10'da olan gerçek eş, kalabalık arttığında 20. veya 30. sıraya düşebilir.

Bu pilotun sorusu şuydu:

> Aynı 1.094 referans çifti, bütün 26 leaf'ten binlerce yeni görüntü eklenince robust top-10
> listesinde kalıyor mu ve aday yükü uygulanabilir düzeyde mi?

Bu çalışma bir **bounded scaled retrieval/capacity pilotudur**. Kanser sınıflandırma modeli
eğitilmedi; frozen duplicate rule çalıştırılmadı; family veya train/validation/test split'i
üretilmedi.

## 2. Kontrollü tasarım

Her stage'de kalibrasyondaki 1.748 görüntünün tamamı korundu. Kalibrasyonda kullanılan dosya
yolları dışlandıktan sonra 26 leaf'in her birinden seed 42 ile iç içe distractor örnekleri seçildi:

| Stage | Yeni distractor/leaf | Yeni distractor | Toplam indeks |
|---:|---:|---:|---:|
| 50 | 50 | 1.300 | 3.048 |
| 100 | 100 | 2.600 | 4.348 |
| 250 | 250 | 6.500 | 8.248 |

`İç içe` örneklem, 50'lik stage'in 100'lük stage'in; 100'lük stage'in de 250'lik stage'in
altkümesi olması demektir. Böylece stage farkı rastgele başka görüntülerden değil, yalnız indeksin
büyümesinden gelir.

V2 kalibrasyonundaki 1.748 descriptor hash doğrulamasıyla yeniden kullanıldı. GPU yalnız 6.500 yeni
görüntü için dört robust ResNet18 descriptor'ı çıkardı. Her descriptor `(512,)` boyutunda ve L2
normalize edildi.

## 3. Retrieval yöntemi

Arama, uyumlu `source_group` içinde dört descriptor kanalının top-k birleşimi olarak yapıldı:

1. ham tam görüntü + flip ortalaması;
2. autocontrast tam görüntü + flip ortalaması;
3. ham multi-crop + flip ortalaması;
4. autocontrast multi-crop + flip ortalaması.

Benzerlik hesabı approximate değil, exact cosine similarity'dir. GPU belleğine bütün matrisi
yüklemek yerine 256 sorguluk bloklar kullanıldı. Aynı seed ve stable-order sıralaması sonucu tekrar
üretilebilir kılar.

Komut:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\benchmark_candidate_generation_scaled.py `
  --stage-sizes 50,100,250 `
  --batch-size 8 `
  --retrieval-block-size 256 `
  --seed 42
```

Önceden belirlenen maksimum-stage kabul koşulları:

- top-10 aynı-kaynak recall: 1.094/1.094;
- doğal aynı-kaynak recall: 30/30;
- top-10 benzersiz aday yükü: en fazla 25 çift/indekslenen görüntü.

## 4. Ana sonuç

| Distractor/leaf | İndeks | Top-k | Aday çift | Çift/görüntü | Aynı-kaynak hit | Recall | Doğal hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 3.048 | 5 | 21.877 | 7,18 | 1.082/1.094 | %98,90 | 30/30 |
| 50 | 3.048 | 10 | 38.144 | 12,51 | 1.092/1.094 | %99,82 | 30/30 |
| 50 | 3.048 | 20 | 74.527 | 24,45 | 1.094/1.094 | %100 | 30/30 |
| 100 | 4.348 | 5 | 32.631 | 7,50 | 1.078/1.094 | %98,54 | 30/30 |
| 100 | 4.348 | 10 | 58.837 | 13,53 | 1.092/1.094 | %99,82 | 30/30 |
| 100 | 4.348 | 20 | 112.965 | 25,98 | 1.094/1.094 | %100 | 30/30 |
| 250 | 8.248 | 5 | 64.082 | 7,77 | 1.075/1.094 | %98,26 | 30/30 |
| **250** | **8.248** | **10** | **122.211** | **14,82** | **1.092/1.094** | **%99,82** | **30/30** |
| 250 | 8.248 | 20 | 234.992 | 28,49 | 1.092/1.094 | %99,82 | 30/30 |

Doğal 30 çiftin tamamı her stage ve her top-k değerinde bulundu. Aday-yükü koşulu da top-10 için
geçildi. Kapının kapanmasının tek nedeni iki kontrollü pozitifin kaçmasıdır.

## 5. Kaçan iki çift

| Pair ID | Dönüşüm | Maksimum stage en iyi rank | En iyi descriptor | Cosine similarity | pHash mesafesi |
|---|---|---:|---|---:|---:|
| C000074 | `brightness_1_20` | 24 | `raw_full_flip_mean` | 0,8704 | 8 |
| C000725 | `brightness_1_20` | 27 | `raw_full_flip_mean` | 0,9137 | 4 |

Bu çiftler 50 ve 100 distractor/leaf stage'lerinde top-20 içindeydi; 250 stage'inde top-20'nin de
dışına düştü. Dolayısıyla yalnız `k=20` seçmek kalıcı çözüm değildir ve aday yükünü maksimum
stage'de 28,49 çift/görüntüye çıkararak kapasite sınırını da aşar.

## 6. Kaynak grubu yükü

Maksimum stage robust top-10 aday yükü:

| Source group | Görüntü | Aday çift | Çift/görüntü |
|---|---:|---:|---:|
| all_blood_smear | 1.264 | 22.199 | 17,56 |
| lymphoma_histology | 942 | 15.682 | 16,65 |
| lung_colon_histology | 1.630 | 26.372 | 16,18 |
| breast_histology | 624 | 10.075 | 16,15 |
| oral_histology | 624 | 9.084 | 14,56 |
| cervix_pap_smear | 1.590 | 21.194 | 13,33 |
| brain_ce_mri | 942 | 11.759 | 12,48 |
| kidney_ct | 632 | 5.846 | 9,25 |

Bu tablo farklı modalitelerde aday yoğunluğunun aynı olmadığını tekrar gösterir. En yüksek yük kan
yayması grubundadır; kaçan iki çift de bu gruptadır.

## 7. Kapasite tahmini

Maksimum stage top-10 değeri 14,817 benzersiz aday çift/görüntüdür. Bu yoğunluk 130.000 görüntüde
de aynı kalırsa yaklaşık 1.926.216 benzersiz aday çifti oluşur.

Frozen duplicate scorer daha önce yaklaşık 0,815 saniye/çift ölçülmüştü. Tek CPU süreciyle kaba
tahmin:

```text
1.926.216 × 0,815 saniye ≈ 18,17 gün
```

Bu lineer tahmin tam veri ölçümü değildir; fakat candidate retrieval hızlı olsa bile frozen rule'ı
her adaya seri uygulamanın hâlâ pahalı olduğunu gösterir. Ucuz safety/priority kanalı ve paralel
skorlama planı gerekecektir.

GPU descriptor çıkarımı 485,14 saniye; üç retrieval stage'inin exact bloklu araması toplam yaklaşık
5,16 saniye sürdü. Bütün pilot 497,91 saniyede tamamlandı. Descriptor çıkarımı baskın maliyettir.

## 8. Post-hoc pHash≤8 safety-channel kontrolü

Robust top-10'un iki kaçırması görüldükten sonra pHash mesafelerinin 8 ve 4 olduğu fark edildi.
Bu nedenle ayrıca tekrar üretilebilir, açıkça **post-hoc exploratory** olarak işaretlenmiş bir
kontrol çalıştırıldı:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\analyze_scaled_phash8_safety.py --radius 8 --workers 4
```

| Kanal | Benzersiz aday |
|---|---:|
| robust top-10 | 122.211 |
| pHash≤8 | 2.831 |
| kesişim | 2.014 |
| birleşim | 123.028 |
| birleşimin eklediği | 817 |

Birleşim 14,92 çift/görüntü üretti ve `C000074` ile `C000725` çiftlerinin ikisini de geri aldı:

- robust top-10: 1.092/1.094;
- robust top-10 ∪ pHash≤8: 1.094/1.094;
- doğal çiftler: 30/30.

Ancak pHash≤8 kanalı iki hatayı gördükten sonra seçildi. Bu nedenle orijinal robust-top-10 pilotunu
geriye dönük olarak “geçti” yapmaz. Yalnız Candidate Generator v2.1 için güçlü bir geliştirme
hipotezidir.

## 9. Bütünlük kontrolleri

- 52/52 Python 3.10 birim testi geçti.
- Görüntü indeksi: 8.248 satır ve 8.248 benzersiz key.
- Kalibrasyon: 1.748; yeni distractor: 6.500; leaf sayısı: 26.
- Pair-stage tablosu: 4.026 = 1.342 × 3 satır.
- Maksimum-stage top-10 manifesti: 122.211 satır ve 122.211 benzersiz indeks çifti.
- Candidate manifestindeki bütün indeks/key eşleşmeleri doğru.
- Dört descriptor matrisi `(8248, 512)` `float32`; NaN/Inf yok ve normlar yaklaşık 1,0.
- Ana pilot ve post-hoc script hash'leri kayıtlı JSON değerleriyle eşleşti.

Yerel artifacts:

- `artifacts/audit/candidate_generation_scaled_v1/report.json`
- `artifacts/audit/candidate_generation_scaled_v1/image_index.csv`
- `artifacts/audit/candidate_generation_scaled_v1/pair_stage_metrics.csv`
- `artifacts/audit/candidate_generation_scaled_v1/max_stage_top10_candidates.csv`
- `artifacts/audit/candidate_generation_scaled_v1/robust_descriptors_max_stage.npz`
- `artifacts/audit/candidate_generation_scaled_v1/posthoc_phash8_safety.json`

## 10. Bilimsel karar

1. Robust top-10 tek başına kalibrasyon ölçeğinde geçmesine rağmen büyüyen indekste iki pozitif
   kaçırdı; **tam veri taramasına kabul edilmedi**.
2. Top-20 de maksimum stage'de aynı iki çifti kaçırdı ve yük sınırını aştı; yalnız k'yi büyütmek
   yeterli değildir.
3. `robust top-10 ∪ pHash≤8`, az ek yükle iki hatayı kurtaran Candidate Generator v2.1 adayıdır.
4. V2.1 aynı açılmış referanslarda seçildiği için yeni, kaynak-görüntüleri ayrılmış bir retrieval
   holdout'unda dondurulup bir kez doğrulanmalıdır.
5. Tam 130.000 görüntü taraması, family oluşturma ve split hâlâ başlamadı.

## 11. Sonraki kontrollü adım

Candidate Generator v2.1 protokolünü **çalıştırmadan önce** dondur:

```text
aday = robust dört-kanal top-10 UNION flip-aware pHash≤8
```

Ardından mevcut kalibrasyon ve 6.500 distractor görüntüsüyle kaynak yolu kesişmeyen yeni bir
kontrollü retrieval holdout'u oluştur. Bütün 26 leaf'ten yeni kaynak görüntüler seç; bilinen
brightness/geometri dönüşümlerini üret; v2.1'i yalnız bir kez değerlendir. Kabul koşulları yine tam
aynı-kaynak/doğal recall ve sınırlı aday yükü olmalıdır.

Bu rapordaki recall, duplicate candidate retrieval ölçüsüdür; kanser sınıflandırma doğruluğu veya
klinik performans değildir.
