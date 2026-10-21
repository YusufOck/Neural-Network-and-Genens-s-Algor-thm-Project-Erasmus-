# Calibration-set v1 development görsel inceleme sonucu

Tarih: 2026-10-21

Durum: **139/139 development çifti AI-destekli görsel incelemeden geçti; 139/139 holdout kararı
boş ve görülmemiş olarak korundu.**

## Kapsam

`review/contact_sheets/` altındaki 32 development PNG sayfasında bulunan 139 çift, Pair ID
üzerinden tek tek incelendi. Diğer 32 PNG sayfasındaki 139 holdout çiftine bakılmadı. Bu inceleme
kanser tanısı değildir; yalnız iki dosyanın aynı kaynak görüntünün dönüşümleri olup olmadığına
ilişkin veri-audit kararıdır.

Kararlar Mehmet'in 2026-10-21 tarihli açık yetkilendirmesiyle Codex tarafından verilmiştir.
`reviewer` alanında `Codex AI-assisted visual review authorized by Mehmet` yazmaktadır. Bu nedenle
sonuç bağımsız uzman-insan gold standardı olarak sunulmamalıdır.

## Kayıt yerleri

- Git ile izlenen karar kaynağı: `research/development_visual_review_v1.csv`
- Güncellenen yerel çalışma tablosu:
  `artifacts/audit/duplicate_calibration_v1/review/review_labels.csv`
- Kararların PNG başlıklarında görüldüğü 32 development sayfası:
  `artifacts/audit/duplicate_calibration_v1/review/reviewed_development_v1/contact_sheets/`
- Güncelleme öncesindeki boş tablo yedeği:
  `artifacts/audit/duplicate_calibration_v1/review/review_labels.before_development_review.csv`
- Güvenli uygulama kodu: `src/apply_calibration_reviews.py`
- Kararlı görsel sayfaları üretme kodu: `src/render_reviewed_contact_sheets.py`

Uygulama kodu yalnız development kararlarını kabul eder; holdout Pair ID verilirse durur. Var olan
farklı bir kararı sessizce değiştirmez ve çalışma tablosunu atomik olarak yazar.

## Sonuçlar

| Karar | Sayı |
|---|---:|
| `same_source_candidate` | 15 |
| `independent_lookalike` | 123 |
| `uncertain` | 1 |
| Toplam | 139 |

Çift üretim türüne göre:

| Aday türü | Aynı kaynak | Bağımsız | Belirsiz | Toplam |
|---|---:|---:|---:|---:|
| Rastgele aynı-alt-sınıf | 0 | 52 | 0 | 52 |
| pHash-hard aynı-alt-sınıf | 15 | 36 | 1 | 52 |
| Uyumlu grupta çapraz-alt-sınıf | 0 | 35 | 0 | 35 |

Bu oranlar veri setindeki doğal duplicate yaygınlığını ölçmez. Kalibrasyon örneklemi belirli aday
türlerinden kontrollü sayıda seçildiği için yalnız yöntem geliştirme kanıtıdır.

## Güçlü aynı-kaynak adayları

`C000587`, `C000591`, `C000603`, `C000604`, `C000605`, `C000606`, `C000607`, `C000609`,
`C000613`, `C000614`, `C000617`, `C000618`, `C000619`, `C000631`, `C000632`.

Görsel kararda genel renk/sınıf benzerliği yerine tekrarlanan nadir hücresel veya anatomik
işaretler arandı. Zor pozitiflerde destekleyici ORB/RANSAC eşleşme sayıları ve bazı CT çiftlerinde
hizalanmış gri-seviye korelasyonu kontrol edildi. Bu destekleyici ölçümler henüz dondurulmuş bir
otomatik karar kuralı veya onaylanmış eşik değildir.

## Belirsiz çift

`C000620`, iki kidney-tumor CT görüntüsünü içerir. Benzer anatomi ve inceleme düzeyi vardır; ancak
farklar tek bir 2B döndürme/kaydırma/ölçek/parlaklık dönüşümüyle güvenle açıklanamamıştır. Aynı
kesitin dönüşümleri ile komşu CT kesitleri ayrımı kesin olmadığı için `uncertain` ve `low`
confidence olarak bırakılmıştır. Belirsizi zorla negatif veya pozitif yapmak yerine ECC/phase
correlation ve sonraki yöntem doğrulamasına taşımak leakage denetimi açısından daha güvenlidir.

## Çıkarım ve sonraki sınır

- Düşük pHash mesafesi tek başına duplicate kanıtı değildir: pHash-hard grubunun 36 çifti bağımsız
  görünmüştür.
- pHash yine yararlı aday üretmiştir: 15 güçlü aynı-kaynak adayı bu kanalda bulunmuştur.
- Random ve çapraz-alt-sınıf development kontrollerinde görsel aynı-kaynak adayı bulunmamıştır.
- Holdout hâlâ kapalıdır. Bir sonraki iş, pHash + ORB/RANSAC + ECC/phase correlation + hizalanmış
  benzerlik kurallarını development kanıtıyla kodlayıp dondurmaktır. Holdout ancak bundan sonra bir
  kez açılıp yöntemin bağımsız ölçümünde kullanılacaktır.
