# Candidate Generator v2.3 Colab/local çalıştırıcı ve throughput shard

Tarih: 2026-11-05

Durum: **Hash-korumalı çalıştırıcı hazırlandı ve yerel RTX 3050 üzerinde 260
görüntülük bounded shard başarıyla tamamlandı. Tam 130.000 görüntülük retrieval,
duplicate kararı, family, split ve model eğitimi başlatılmadı.**

## 1. Bu adımın amacı

Bu çalışma bir kanser sınıflandırma deneyi değildir. Amaç, dondurulmuş Candidate
Generator v2.3 yönteminin tam taramadan önce şu teknik zincirini sınamaktır:

1. frozen config ve kaynak kod hash'lerini doğrulamak;
2. ResNet18 checkpoint hash'ini doğrulamak;
3. veri setinin hâlâ 130.000 görüntü, 26 leaf ve 8 source group olduğunu kontrol etmek;
4. her leaf'ten eşit ve deterministik küçük bir örnek seçmek;
5. iki flip-aware pHash ile dört robust ResNet18 descriptor kanalını üretmek;
6. source-group içinde robust top-15 ile pHash `<=12` adaylarını birleştirmek;
7. çıktı dosyalarını SHA-256 manifestiyle kaydetmek.

## 2. Çalıştırılan kapsam

| Parametre | Değer |
|---|---:|
| Leaf sayısı | 26 |
| Her leaf'ten görüntü | 10 |
| Toplam shard görüntüsü | 260 |
| Seed | 42 |
| Robust komşu sayısı | top-15 |
| pHash yarıçapı | `<=12` |
| Arama alanı | yalnız uyumlu source group içi |
| Batch size | 8 |

Tam veri envanteri shard başlamadan önce tekrar doğrulandı: 130.000 görüntü,
26 leaf × 5.000 ve 8 source group.

## 3. Yerel ortam

| Bileşen | Değer |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.5.1+cu118 |
| torchvision | 0.20.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Frozen config SHA-256 | `daef2f4d1c4199846a64b549a5070f5c8cc3f322cc949fb55dde2700ef2e3776` |
| ResNet18 checkpoint SHA-256 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

## 4. Ölçülen sonuç

| Ölçüm | Sonuç |
|---|---:|
| Source-group içi mümkün çift | 4.670 |
| Robust top-15 adayı | 3.168 |
| pHash `<=12` adayı | 8 |
| İki kanalda ortak | 7 |
| Tekrarsız birleşim | 3.169 |
| Birleşim çifti / görüntü | 12,188 |
| Source-group dışı aday | 0 |
| pHash üretimi | 0,637 saniye |
| Robust descriptor üretimi | 17,139 saniye |
| Robust retrieval | 0,106 saniye |
| Toplam, rapor yazımından önce | 18,048 saniye |
| Descriptor hızı | 15,170 görüntü/saniye |

`12,188 çift/görüntü`, frozen güvenlik sınırı olan 25'in altındadır. Ancak küçük
source-group'larda top-15 aday yoğunluğu doygunluğa uğradığı için bu shard değeri
130.000 görüntüye lineer süre veya aday sayısı tahmini olarak kullanılamaz. Önceki
yaklaşık 2,93 milyon full-run kapasite değeri de hâlâ tahmindir.

## 5. Üretilen ve doğrulanan çıktılar

Yerel çıktı klasörü:
`artifacts/audit/candidate_generator_v2_3_throughput_shard/`

- `image_index.csv`: 260 deterministik görüntü satırı;
- `phash_variants.npz`: `original` ve `horizontal_flip`, her biri `(260,)`;
- `robust_descriptors.npz`: dört frozen descriptor, her biri `(260, 512)`;
- `candidate_pairs.csv`: 3.169 canonical aday çifti;
- `run_report.json`: yöntem, ortam, süre, sayım ve bütünlük bilgileri;
- `hash_manifest.json`: diğer beş çıktının SHA-256 değerleri.

Manifestteki bütün hash'ler yeniden hesaplanarak eşleşti. Adayların tamamında
`left_index < right_index`; tekrarsız çift sayısı 3.169 ve source-group dışı çift
sayısı 0'dır. Notebook JSON doğrulaması, yeni shard birim testleri ve bütün proje
test paketi geçti: **88/88 test başarılı**.

## 6. Colab/local notebook

Notebook:
`notebooks/candidate_generator_v2_3_colab_local.ipynb`

Notebook aynı script'i çalıştırır; yöntemi ayrı bir Colab sürümüyle yeniden yazmaz.
Bu sayede local ve Colab arasında tek uygulama kaynağı korunur. Colab'da yalnız veri
seti ve kalıcı output yolları kullanıcı tarafından ayarlanır. Başarılı çalışmadan
sonra Codex'e en az şu iki dosya geri verilmelidir:

- `run_report.json`;
- `hash_manifest.json`.

Tam inceleme gerekirse indeks, pHash, descriptor ve candidate CSV/NPZ dosyaları da
aynı output klasöründe korunmalıdır.

## 7. Bu sonuç neyi kanıtlamaz?

- 3.169 adayın 3.169'u duplicate değildir; bunlar yalnız ayrıntılı doğrulamaya
  gönderilecek kısa-listedeki çiftlerdir.
- Bu bir accuracy, macro-F1 veya model performansı değildir.
- Küçük-shard descriptor hızı tam tarama süresi garantisi değildir.
- Hasta kimliği bulunmadığı için patient-level leakage çözüldü denemez.
- Family, train/validation/test split veya classifier eğitimi yapılmadı.

## 8. Sıradaki karar noktası

Altyapı smoke testi geçti. Bir sonraki teknik aşama tam retrieval çalıştırıcısını
parçalı, hash-doğrulamalı ve kaldığı yerden devam edebilir biçimde hazırlamaktır.
Tam 130.000 görüntülük retrieval yalnız bu kod ve plan ayrıca test edildikten ve
Mehmet açıkça onay verdikten sonra başlatılmalıdır.
