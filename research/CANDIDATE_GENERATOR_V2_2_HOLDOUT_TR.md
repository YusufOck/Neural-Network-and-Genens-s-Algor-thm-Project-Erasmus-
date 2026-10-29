# Candidate Generator v2.2 ikinci kaynak-ayrık kontrollü holdout değerlendirmesi

Tarih: 2026-10-29

Durum: **Önceden dondurulan kabul kapısı geçilmedi.** V2.2, yeni kontrollü
holdout'taki 1.040 aynı-kaynak çiftinin 1.039'unu buldu. Aday yükü sınırı geçti;
ancak toplam ve her-dönüşüm recall koşulu bir `brightness_1_20` kaçağı nedeniyle
geçmedi. Tam 130.000 görüntü taraması, family oluşturma, split ve model eğitimi
başlatılmadı.

## 1. Deneyin sorusu

V2.1'in tek `combined_stress` kaçağı en iyi robust rank 13'te kalmıştı. V2.1
sonucu görüldükten sonraki tanısal kontrolde robust `top-15` bu çifti geri almış,
aday yükü de 25 çift/görüntü sınırının altında kalmıştı. Bu nedenle, ikinci yeni
holdout açılmadan önce aşağıdaki v2.2 hipotezi config ve SHA-256 ile donduruldu:

```text
candidate = robust four-channel directed top-15 UNION flip-aware pHash<=8
```

Bu deney kanser sınıflandırması değildir. Yalnızca aynı kaynak görüntüden üretilen
kontrollü bir akrabanın pahalı duplicate doğrulamasına gönderilecek kısa listeye
girip girmediğini ölçer.

## 2. Önceden dondurulan protokol

- Config: `config/candidate_generator_v2_2.json`
- Frozen SHA-256:
  `85c5d0791439694b96a84b155a2d0fe95379dc2b2ce04a7369d5061b25a29953`
- Sonucu görmeden önceki Git commit'i: `e579589`
- Arama yalnız uyumlu sekiz `source_group` içinde yapıldı.
- Robust kanal: dört ImageNet-ön-eğitimli ResNet18 descriptor kanalının yönlü
  top-15 birleşimi.
- Güvenlik kanalı: orijinal/yatay-çevrilmiş 64-bit pHash için Hamming mesafesi
  `<=8`.
- Kabul kapısı:
  - toplam kontrollü aynı-kaynak recall = %100;
  - on dönüşümün her birinde recall = %100;
  - en fazla 25 benzersiz aday çift/görüntü.

Config, build/evaluation kodu ve ilgili testler holdout değerlendirilmeden önce
56/56 Python 3.10 testi geçti ve GitHub `main` dalına push edildi.

## 3. İkinci holdout kapsamı

26 leaf'in her birinden deterministik olarak 4 yeni kaynak görüntü, her kaynak
için 10 kontrollü dönüşüm ve 250 yeni etiketsiz distractor seçildi.

| Bileşen | Adet |
|---|---:|
| Kaynak orijinal | 104 |
| Kontrollü dönüşüm | 1.040 |
| Etiketsiz distractor | 6.500 |
| İndekslenen benzersiz görüntü | 7.644 |
| Bilinen aynı-kaynak çifti | 1.040 |
| Leaf | 26/26 |
| Source group | 8/8 |

Bu holdout; scaled pilotta kullanılan 7.208 dataset yolunu, v2.1 holdout'taki
6.604 dataset yolunu ve bilinen exact-duplicate gruplarındaki bütün üyeleri
dışladı. Önceki iki retrieval kapsamıyla birleşik olarak 13.812 dataset yolu
dışlandı. Yol ve exact-içerik ayrıklığı hasta/slide bağımsızlığını kanıtlamaz;
dataset bu kimlikleri sağlamamaktadır.

## 4. Ana sonuç

| Kanal | Aday çift | Kontrollü hit | Recall |
|---|---:|---:|---:|
| Robust dört-kanal top-15 | 165.713 | 1.036/1.040 | %99,615 |
| Flip-aware pHash<=8 | 2.515 | 472/1.040 | %45,385 |
| V2.2 birleşim | 166.372 | 1.039/1.040 | **%99,904** |

- İki kanalın kesişimi: 1.856 çift.
- pHash kanalının robust top-15 üzerine net eklediği: 659 çift.
- Aday yükü: **21,765 çift/görüntü**.
- Kaynak grubu içinde mümkün 4.145.106 çiftin %4,014'ü shortlist edildi.
- Yük koşulu `21,765 <= 25` ile geçti.
- Recall koşulu `1.039/1.040 < 1.040/1.040` nedeniyle geçmedi.

Frozen sonuç geriye dönük olarak değiştirilemez:

```text
second_prospective_controlled_holdout_failed
```

## 5. Dönüşüm bazında sonuç

| Dönüşüm | Pozitif | Robust hit | pHash hit | V2.2 hit |
|---|---:|---:|---:|---:|
| brightness_0_20 | 104 | 104 | 104 | 104 |
| brightness_1_20 | 104 | 100 | 103 | **103** |
| combined_stress | 104 | 104 | 0 | 104 |
| horizontal_flip | 104 | 104 | 104 | 104 |
| rotate_minus_10 | 104 | 104 | 12 | 104 |
| rotate_plus_10 | 104 | 104 | 14 | 104 |
| shear_factor_plus_0_10 | 104 | 104 | 79 | 104 |
| shift_plus10_minus10 | 104 | 104 | 0 | 104 |
| zoom_in_1_10 | 104 | 104 | 43 | 104 |
| zoom_out_0_90 | 104 | 104 | 13 | 104 |

pHash tek başına geometrik dönüşümlerde yeterli değildir. V2.2'de robust kanalın
kaçırdığı dört parlaklık çiftinin üçünü pHash geri aldı; kalan bir çift her iki
kanalın da sınırının dışında kaldı.

## 6. Kaçan tek çift

| Alan | Değer |
|---|---|
| Pair ID | `V22H00149` |
| Leaf | `all_pro` |
| Source group | `all_blood_smear` |
| Kaynak | `ALL/all_pro/all_pro_4233.jpg` |
| Dönüşüm | `brightness_1_20` |
| pHash mesafesi | 12 |

Bu pozitif insan yorumuna dayanmaz: sağ görüntü doğrudan sol görüntüye 1,20
parlaklık dönüşümü uygulanarak üretildi. Görsel incelemede hücrelerin konumları
aynıdır; parlak bölgelerde ciddi beyaz doyumu vardır.

| Descriptor | Cosine similarity | Sol->sağ rank | Sağ->sol rank | En iyi rank |
|---|---:|---:|---:|---:|
| raw full flip mean | 0,869679 | 524 | 58 | 58 |
| autocontrast full flip mean | 0,851524 | 847 | 106 | 106 |
| raw multicrop flip mean | 0,833914 | 909 | 52 | **52** |
| autocontrast multicrop flip mean | 0,845586 | 921 | 166 | 166 |

Gerçek eş hiçbir descriptor/yönde ilk 15'e giremedi; en iyi rank 52'dir. pHash
mesafesi 12 olduğu için `pHash<=8` kanalına da giremedi. Sonuç bir kod çökmesi
değil, dondurulmuş retrieval kuralının ölçülmüş false-negative'idir.

## 7. Sonuç görüldükten sonraki tanısal kontrol

Aşağıdaki analiz **post-hoc/exploratory** çalışmadır; v2.2 sonucunu başarılıya
çevirmez ve aynı holdout yeniden test sayılmaz.

| Robust top-15 + pHash yarıçapı | Birleşim aday | Çift/görüntü | Hit |
|---:|---:|---:|---:|
| 8 | 166.372 | 21,765 | 1.039/1.040 |
| 9 | 166.372 | 21,765 | 1.039/1.040 |
| 10 | 168.014 | 21,980 | 1.039/1.040 |
| 11 | 168.014 | 21,980 | 1.039/1.040 |
| 12 | 172.000 | 22,501 | 1.040/1.040 |

`top-15 union pHash<=12`, bu holdout'ta tam kontrollü recall ve mevcut yük
bütçesini birlikte sağlamıştır. Fakat bu kural kaçak görüldükten sonra seçildiği
için yalnızca olası v2.3 hipotezidir. Bağımsız yeni kanıt olmadan kabul edilemez.

## 8. Süre ve bütünlük kontrolleri

- pHash: 23,60 saniye.
- Dört robust descriptor: 549,90 saniye.
- Exact bloklu retrieval: 2,37 saniye.
- Toplam: 581,64 saniye (yaklaşık 9 dakika 42 saniye).
- Descriptor ve retrieval cihazı: CUDA.
- Dört descriptor matrisi de `7644 x 512`, sonlu ve birim norma sahiptir.
- Candidate manifest: 166.372 satır ve 166.372 benzersiz çift.
- Manifestte bilinen pozitif: 1.039.
- Config sidecar SHA-256 değerlendirme sonrasında da eşleşmektedir.
- ResNet18 checkpoint SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

Ham çıktılar Git'e eklenmemiştir; yerel olarak
`artifacts/audit/candidate_generator_v2_2_holdout/evaluation/` altındadır.

## 9. Bilimsel karar ve önerilen sonraki kontrol noktası

1. V2.2 uygun aday yüküne karşın önceden ilan edilen tam-recall kapısını bir
   çiftle kaçırdı; **tam veri taramasına kabul edilmedi**.
2. Holdout tekrar çalıştırılmayacak ve sonuç geriye dönük değiştirilmeyecek.
3. `top-15 union pHash<=12` yalnızca v2.3 geliştirme hipotezidir.
4. Üçüncü kontrollü holdout üretmeye geçmeden önce art arda sentetik holdout'a
   uyum sağlama riskini değerlendirmek gerekir. Kontrollü dönüşümler doğal yakın
   kopya recall'ını veya aday precision'ını ölçmez.
5. Full scan, family, leakage-aware split ve model eğitimi hâlâ sonraki
   aşamalardır; bunların hiçbiri bu deney sırasında başlatılmadı.

Bu rapordaki %99,904 değerinin kanser sınıflandırma accuracy, macro-F1 veya
klinik performansla ilgisi yoktur.
