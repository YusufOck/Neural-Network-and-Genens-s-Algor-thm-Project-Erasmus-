# Natural near-duplicate audit v1 sonucu

Tarih: 2026-11-04

Durum: **Score-blind görsel kararlar private seçim katmanlarıyla açıldı ve
dondurulmuş duplicate rule v1 ile 345/345 çift yerelde karşılaştırıldı.** Bu bir
veri sızıntısı denetimidir; kanser modeli eğitimi, duplicate family üretimi veya
train/validation/test split değildir.

## 1. Giriş bütünlüğü

| Kaynak | SHA-256 |
|---|---|
| Private seçim manifesti | `dc01803890352140c9e3465b86d144082850ebff735b73adc4f40a1ea97b86f0` |
| Dondurulmuş score-blind görsel kararlar | `91d857685ada7dbd6d3fb6e3add5a8fe914e3767389f1d37401a660ffd308bca` |
| Frozen duplicate rule v1 | `5ffa80daa6e6ab0eb5f3ac179073de2f0f018325310310b33adc4c137710b645` |
| Frozen development scorer | `ac9f0a1538a24615bea6fac530806c1536aefb3ad0916fd3f2895d8ca4af513f` |

Görsel kararlar skorlar açılmadan önce ayrı commit ile donduruldu. Analiz kodu
Pair ID kapsamını, satır sayısını ve giriş hash'lerini işlemden önce doğruladı.

## 2. Candidate/outside sonucu

| Seçim katmanı | Aynı-kaynak adayı | Bağımsız-benzer | Belirsiz | Toplam |
|---|---:|---:|---:|---:|
| `candidate_phash_supported` | 21 | 35 | 1 | 57 |
| `candidate_robust_only` | 0 | 95 | 1 | 96 |
| `outside_phash_distance14_16` | 0 | 64 | 0 | 64 |
| `outside_robust_rank16_30` | 0 | 64 | 0 | 64 |
| `outside_random` | 0 | 64 | 0 | 64 |
| **Toplam** | **21** | **322** | **2** | **345** |

Sonuç:

- Görsel olarak aynı-kaynak kabul edilen 21 çiftin tamamı v2.3 candidate
  bölgesindedir.
- Candidate bölgesindeki gözlenen pozitif verim 21/153 = %13,73'tür; iki
  belirsiz çıkarıldığında 21/151 = %13,91'dir.
- Önceden belirlenen üç `outside_*` challenge katmanındaki 192 çiftin hiçbirinde
  doğrulanmış aynı-kaynak çift bulunmadı.
- Bu nedenle **önceden belirlenen natural challenge-set kapısı geçti.**

Bu sonuç, 3.093.466 olası doğal çiftin tamamında recall'ın %100 olduğunu
kanıtlamaz. Yalnızca incelenen, önceden dondurulmuş challenge set için kanıttır.

## 3. Frozen duplicate rule v1 karşılaştırması

345 çift 256×256 analiz boyutunda; ORB/RANSAC, phase correlation, uygun
modalitelerde ECC ve hizalanmış SSIM/NMI/korelasyon kanıtlarıyla skorlandı. Kural
eşikleri değiştirilmedi. İşlem 200,4 saniye sürdü.

Görsel referansı `uncertain` olan iki çift metrik hesabından çıkarıldığında:

| Ölçüm | Sonuç |
|---|---:|
| Değerlendirilen referans | 343 |
| Görsel aynı-kaynak | 21 |
| Aynı-kaynak yakalama | 21/21 = %100 |
| Görsel bağımsız | 322 |
| Kesin bağımsız kararı | 299 |
| Güvenli `uncertain` kararı | 21 |
| Görsel bağımsızken otomatik aynı-kaynak | 2 |
| Karar kapsamı | %93,88 |
| Kesin kararlardaki doğruluk | %99,38 |

Buradaki %100 yakalama yalnız 21 gözlenmiş doğal pozitif içindir; bütün veri seti
için garantili recall değildir. Ayrıca görsel referans AI-desteklidir ve bağımsız
klinik uzman gold standard değildir.

## 4. Uyuşmazlıklar ve güvenli konsensüs

Otomatik family üretiminde tek başına kural veya tek başına görsel karar
kullanılmayacaktır:

| Konsensüs | Çift sayısı | Kullanım |
|---|---:|---|
| Görsel aynı-kaynak + kural aynı-kaynak | 21 | Aynı-kaynak konsensüsü |
| Görsel bağımsız + kural bağımsız | 299 | Bağımsız konsensüsü |
| Kararlardan biri belirsiz veya kararlar çatışıyor | 25 | Manuel/adjudication kuyruğu |

İki doğrudan çatışma:

- `N000123`, Pap smear: görsel karar `independent_lookalike`, kural
  `strong_intensity_structure` yoluyla `same_source`. ORB ile hizalanmış
  SSIM `0,8264`, NMI `1,3370`, korelasyon `0,9577`. Yeniden görsel kontrolde
  birebir yapısal eşleşme ihtimali görüldü; dondurulmuş karar geriye dönük
  değiştirilmedi ve adjudication'a bırakıldı.
- `N000206`, kidney CT: görsel karar `independent_lookalike`, kural
  `strong_geometry` yoluyla `same_source`. 309 ORB inlier, inlier ratio
  `0,5712`, minimum kapsama `0,5038`. Kesitler çok benzer olsa da komşu/farklı
  CT kesitleri olma ihtimali nedeniyle otomatik aynı-görüntü ailesine
  bağlanmayacaktır.

Görsel olarak belirsiz iki çift:

- `N000252`: kural da `uncertain` verdi.
- `N000337`: kural `independent` verdi; görsel belirsizlik nedeniyle yine
  adjudication kuyruğunda kalır.

## 5. Çıktılar

- Açılmış çift tablosu:
  `research/natural_duplicate_audit_v1/unblinded_pair_results.csv`
- Açılmış özet:
  `research/natural_duplicate_audit_v1/unblinded_report.json`
- Ham skor/karar tablosu (yerel artefakt):
  `artifacts/audit/natural_duplicate_rule_evaluation_v1/pair_scores_and_decisions.csv`
- Ham değerlendirme raporu (yerel artefakt):
  `artifacts/audit/natural_duplicate_rule_evaluation_v1/report.json`

Ham skor CSV SHA-256:
`76d8e739a9a59b26e7ef8de44bdb534e5f8be06b90d748dbfb7774e8c62df3da`.
Ham rapor SHA-256:
`7c2499fee464b848a0c242233ff8605388ff789d6569b5b0c48911aee9a280cc`.

## 6. Sonraki kontrollü adım

1. 25 çiftlik konsensüs dışı kuyruğu izlenebilir bir adjudication tablosuna
   dönüştürmek.
2. Özellikle `N000123` ve `N000206` için “aynı 2B görüntünün dönüşümü” ile
   “benzer/komşu tıbbi kesit” ayrımını korumak.
3. Ardından Candidate Generator v2.3'ü dondurup 130.000 görüntü taramasının
   süre/depolama planını hazırlamak.
4. Full candidate retrieval GPU/Colab'a taşınabilir; milyonlarca adayın
   ORB/ECC doğrulaması CPU ağırlıklı olduğu için ayrıca paralel ve aşamalı bir
   doğrulama planı gerektirir.

Bu adımlar tamamlanmadan duplicate family, leakage-aware split ve final model
eğitimi başlamaz.
