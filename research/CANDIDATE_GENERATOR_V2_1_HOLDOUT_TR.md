# Candidate Generator v2.1 kaynak-ayrık kontrollü holdout değerlendirmesi

Tarih: 2026-10-28

Durum: **Önceden dondurulan kabul kapısı geçilmedi.** V2.1 yeni kontrollü
holdout'taki 1.040 aynı-kaynak çiftinin 1.039'unu buldu. Aday yükü sınırı geçti,
ancak toplam ve her-dönüşüm recall koşulu bir `combined_stress` kaçağı nedeniyle
geçmedi. Tam 130.000 görüntü taraması başlatılmadı.

## 1. Bu deneyin sorusu

Önceki 8.248 görüntülük scaled pilotta robust top-10 iki `brightness_1_20`
pozitifini kaçırmıştı. Sonradan yapılan keşifsel kontrolde flip-aware pHash≤8
kanalı bu iki çifti az ek yükle geri almıştı. Bu nedenle aşağıdaki v2.1 hipotezi,
yeni holdout sonucu görülmeden önce donduruldu:

```text
candidate = robust four-channel top-10 UNION flip-aware pHash<=8
```

Bu deney kanser sınıflandırma modeli değildir. Soru yalnız şudur: aynı kaynak
görüntünün kontrollü olarak değiştirilmiş hali, pahalı duplicate karar kuralına
gönderilecek kısa aday listesine giriyor mu?

## 2. Önceden dondurulan protokol

- Config: `config/candidate_generator_v2_1.json`
- Frozen SHA-256:
  `c689e7efa77cad2fbf96fc4f7c3d1331409719af8a6042b12cd8a6e92a492ef0`
- Sonucu görmeden önceki Git commit'i: `7aa38a9`
- Arama yalnız uyumlu sekiz `source_group` içinde yapıldı.
- Robust kanal: dört ResNet18 descriptor kanalının directed top-10 birleşimi.
- Güvenlik kanalı: orijinal/yatay-çevrilmiş 64-bit pHash için Hamming mesafesi ≤8.
- Kabul kapısı:
  - toplam kontrollü aynı-kaynak recall = %100;
  - on dönüşümün her birinde recall = %100;
  - en fazla 25 benzersiz aday çift/görüntü.

Config ve iki çalıştırıcı, holdout değerlendirilmeden önce 54/54 Python 3.10
testinden geçti ve GitHub `main` dalına push edildi.

## 3. Yeni holdout nasıl kuruldu?

26 leaf'in her birinden deterministik olarak:

- 4 yeni kaynak görüntü;
- her kaynak için 10 kontrollü dönüşüm;
- 250 yeni, etiketsiz distractor seçildi.

Toplam kapsam:

| Bileşen | Adet |
|---|---:|
| Kaynak orijinal | 104 |
| Kontrollü dönüşüm | 1.040 |
| Etiketsiz distractor | 6.500 |
| İndekslenen benzersiz görüntü | 7.644 |
| Bilinen aynı-kaynak çifti | 1.040 |
| Leaf | 26/26 |
| Source group | 8/8 |

Her leaf'te aynı dağılım vardır: 4 kaynak, 40 kontrollü dönüşüm ve 250
distractor. Önceki scaled pilot indeksindeki 7.208 dataset yolu dışlandı.
Ayrıca tam 130.000 görüntülük SHA-256 denetiminde bulunan 2.673 exact-duplicate
grup üyesinin tamamı dışlandı.

Bağımsız doğrulama:

- image index satırı: 7.644;
- benzersiz image key: 7.644;
- üretilen dosya: 1.040;
- positive pair satırı: 1.040;
- önceki dataset yolu kesişimi: 0;
- exact-duplicate grup üyesi kesişimi: 0.

Bu kontroller patient independence kanıtlamaz. Yerel veri setinde hasta/slide
kimliği ve eksiksiz provenance yoktur.

## 4. Ana sonuç

| Kanal | Aday çift | Kontrollü hit | Recall |
|---|---:|---:|---:|
| Robust dört-kanal top-10 | 113.352 | 1.033/1.040 | %99,327 |
| Flip-aware pHash≤8 | 3.191 | 479/1.040 | %46,058 |
| V2.1 birleşim | 114.590 | 1.039/1.040 | **%99,904** |

- İki kanalın kesişimi: 1.953 aday çift.
- pHash kanalının robust top-10 üzerine net eklediği: 1.238 aday çift.
- V2.1 aday yükü: **14,991 çift/görüntü**.
- Kaynak grubu içinde mümkün olan 4.145.106 çiftin %2,764'ü shortlist edildi.
- Yük koşulu `14,991 <= 25` ile geçti.
- Recall koşulu `1.039/1.040 < 1.040/1.040` nedeniyle geçmedi.

Dolayısıyla frozen değerlendirme durumu:

```text
prospective_controlled_holdout_failed
```

## 5. Dönüşüm bazında sonuç

| Dönüşüm | Pozitif | Robust hit | pHash hit | V2.1 hit |
|---|---:|---:|---:|---:|
| brightness_0_20 | 104 | 104 | 104 | 104 |
| brightness_1_20 | 104 | 98 | 104 | 104 |
| combined_stress | 104 | 103 | 0 | **103** |
| horizontal_flip | 104 | 104 | 104 | 104 |
| rotate_minus_10 | 104 | 104 | 10 | 104 |
| rotate_plus_10 | 104 | 104 | 13 | 104 |
| shear_factor_plus_0_10 | 104 | 104 | 80 | 104 |
| shift_plus10_minus10 | 104 | 104 | 0 | 104 |
| zoom_in_1_10 | 104 | 104 | 46 | 104 |
| zoom_out_0_90 | 104 | 104 | 18 | 104 |

pHash tek başına geometrik dönüşümlerde zayıftır; görevi bütün pozitifleri bulmak
değil, robust retrieval'ın özellikle parlaklık değişiminde bıraktığı boşluğu ucuz
bir kanalla kapatmaktır. V2.1, robust kanalın altı `brightness_1_20` kaçağının
tamamını geri aldı.

## 6. Kaçan tek çift

| Alan | Değer |
|---|---|
| Pair ID | `V21H00130` |
| Leaf | `all_pro` |
| Source group | `all_blood_smear` |
| Kaynak | `ALL/all_pro/all_pro_1363.jpg` |
| Dönüşüm | `combined_stress` |
| pHash mesafesi | 22 |

`combined_stress`, aynı anda +7° rotasyon, %8 yatay/dikey kaydırma, 0,08
shear, 1,08 zoom, yatay flip ve 0,70 brightness uygular. Görsel kontrolde
orijinal ve dönüşümün aynı hücre düzeninden geldiği açıktır; ground truth zaten
dosyanın doğrudan bu orijinalden üretilmesine dayanır.

Robust kanal tanısı:

| Descriptor | Cosine similarity | Sol→sağ rank | Sağ→sol rank | En iyi rank |
|---|---:|---:|---:|---:|
| raw full flip mean | 0,8660 | 698 | 25 | 25 |
| autocontrast full flip mean | 0,9554 | 113 | 23 | 23 |
| raw multicrop flip mean | 0,8937 | 529 | 149 | 149 |
| autocontrast multicrop flip mean | 0,9729 | 137 | 13 | **13** |

En iyi yönlü rank 13 olduğu için top-10 shortlist'e giremedi. pHash mesafesi
22 olduğu için pHash≤8 kanalına da giremedi.

## 7. Sonuç görüldükten sonraki tanısal kontrol

Aşağıdaki tablo **post-hoc exploratory** çalışmadır; frozen v2.1 sonucunu
değiştirmez:

| Robust k + pHash≤8 | Birleşim aday | Çift/görüntü | Hit |
|---:|---:|---:|---:|
| 10 | 114.590 | 14,991 | 1.039/1.040 |
| 11 | 125.073 | 16,362 | 1.039/1.040 |
| 12 | 135.784 | 17,763 | 1.039/1.040 |
| 13 | 146.284 | 19,137 | 1.040/1.040 |
| 15 | 167.207 | 21,874 | 1.040/1.040 |
| 20 | 217.930 | 28,510 | 1.040/1.040 |

Top-13 ilk kurtaran değerdir. Top-15, küçük bir rank marjıyla hâlâ mevcut 25
çift/görüntü bütçesinin altındadır. Top-20 ise bu holdout'ta bütçeyi aşar.
Bu nedenle `robust top-15 UNION pHash≤8`, Candidate Generator v2.2 için makul
bir **geliştirme hipotezidir**; doğrulanmış sonuç değildir.

## 8. Zaman ve bütünlük

- pHash hesaplama/adayı: 19,99 saniye.
- Exact bloklu robust retrieval: 2,31 saniye.
- Descriptor çıkarımı dahil toplam: 497,09 saniye.
- CUDA cihazı kullanıldı.
- ResNet18 checkpoint SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Candidate manifest: 114.590 satır ve 114.590 benzersiz çift.
- Candidate manifestte bulunan known-positive satırı: 1.039.
- Config dosyası ve sidecar hash'i değerlendirme sonrasında da eşleşti.

Yerel, Git tarafından izlenmeyen ham artifacts:

- `artifacts/audit/candidate_generator_v2_1_holdout/build_report.json`
- `artifacts/audit/candidate_generator_v2_1_holdout/image_index.csv`
- `artifacts/audit/candidate_generator_v2_1_holdout/positive_pairs.csv`
- `artifacts/audit/candidate_generator_v2_1_holdout/evaluation/report.json`
- `artifacts/audit/candidate_generator_v2_1_holdout/evaluation/positive_pair_metrics.csv`
- `artifacts/audit/candidate_generator_v2_1_holdout/evaluation/candidate_pairs.csv`
- `artifacts/audit/candidate_generator_v2_1_holdout/evaluation/robust_descriptors.npz`
- `artifacts/audit/candidate_generator_v2_1_holdout/evaluation/phash_variants.npz`

## 9. Bilimsel karar ve sonraki adım

1. Candidate Generator v2.1 çok yüksek kontrollü recall ve uygun yük sağladı,
   fakat önceden belirlenen tam-recall kapısını bir çiftle kaçırdı; **tam veri
   taramasına kabul edilmedi**.
2. V2.1 sonucu geriye dönük değiştirilmez ve yeniden çalıştırılmaz.
3. Bu holdout artık development kanıtıdır. `top-15 ∪ pHash≤8` v2.2 hipotezi
   olarak config/sha256 ile dondurulabilir.
4. V2.2, önceki kalibrasyon, scaled pilot ve bu v2.1 holdout'undaki bütün dataset
   yollarını dışlayan ikinci yeni kaynak-ayrık kontrollü holdout'ta yalnız bir kez
   değerlendirilmelidir.
5. V2.2 geçse bile kontrollü dönüşüm recall'ı doğal near-duplicate recall veya
   precision değildir. Tam tarama öncesi doğal/provenance belirsizliği ve scorer
   kapasitesi ayrıca ele alınmalıdır.

Bu sonuç kanser sınıflandırma accuracy, macro-F1 veya klinik performans değildir.
