# Candidate Generator v2.5 — Score-Blind Render Sonucu

Tarih: 2026-11-10

İşlem türü: Score-blind review altyapısı / render audit'i

Durum: Temas sayfaları üretildi ve doğrulandı; görsel kararlar henüz başlamadı

## 1. Amaç

Bu adımın amacı, dondurulmuş 256 prospective çifti private seçim bilgilerini göstermeden insan
gözüyle incelenebilir temas sayfalarına dönüştürmektir. Bu işlem duplicate kararı vermez; yalnız
inceleme arayüzünü hazırlar.

Renderer yalnız şu iki public girdiyi okur:

- `selection_report.json`
- `review_manifest.csv`

`selection_private.csv` okunmaz. Sayfalarda yalnız Pair ID, source group, sol/sağ işareti ve iki
görüntü gösterilir. pHash, robust rank, stratum ve candidate üyeliği gizli kalır.

## 2. Doğrulanmış çıktı

| Kontrol | Sonuç |
|---|---:|
| Review çifti | 256 |
| Benzersiz görüntü | 512 |
| Sayfa başına çift | 6 |
| Temas sayfası | 43 |
| Index satırı / benzersiz Pair ID | 256 / 256 |
| Boş review kararı | 256 / 256 |
| Render bütünlük kontrolü | 12 / 12 geçti |
| Tam proje test paketi | 131 / 131 geçti |

İlk sayfa yalnız layout QA amacıyla açıldı. Başlık, sayfa numarası, `SCORE-BLIND` işareti, Pair
ID'ler, source group ve L/R göstergeleri okunaklıydı; kırpılma veya private bilgi sızıntısı yoktu.
Görüntüler için duplicate/independent/uncertain kararı verilmedi.

## 3. Bütünlük zinciri

| Artifact | SHA-256 |
|---|---|
| Renderer | `f09e8f9dcaadcce169e03576f3507c9ca0458d65bd904fa5e6304255e8fee35d` |
| `render_report.json` | `6266a6904afd188f47df7d24b1aec4919c5165fa38d1e3237b115a71648953a2` |
| `sheet_index.csv` | `6c580f052595bd8aaefef893ab57c78aade9f10e36b4e6c6683c2b6ada25a53f` |
| Frozen public review manifest | `b551ef6bea86ee62c9fc831a48dd0b662a99344f25da7ec9ed550a2a27103bfb` |

`render_report.json`, 43 sayfanın her biri için ayrı SHA-256 değeri taşır. PNG dosyaları büyük ve
yeniden üretilebilir oldukları için Git'e eklenmedi; yerel artifact klasöründe tutulur:

`artifacts/audit/candidate_generator_v2_5_prospective_review_v1/contact_sheets/`

Kalıcı ve küçük metadata kopyaları:

`research/candidate_generator_v2_5_prospective_review_v1/`

## 4. Bilimsel anlamı

Bu sonuç, score-blind görsel inceleme paketinin doğru ve izlenebilir biçimde hazır olduğunu
kanıtlar. V2.5'in near-duplicate recall kapısını geçtiğini kanıtlamaz. Henüz:

- hiçbir görsel karar kaydedilmedi;
- private stratum/skor bilgileri açılmadı;
- candidate scoring veya family construction yapılmadı;
- train/validation/test split hazırlanmadı;
- model eğitilmedi.

## 5. Sonraki kontrollü adım

Mehmet'in ayrı onayından sonra 43 score-blind temas sayfasındaki 256 çift incelenecek ve yalnız
public Pair ID üzerinden `same_source_candidate`, `independent_lookalike` veya `uncertain` kararı
kaydedilecek. Bütün kararlar dondurulmadan private manifest açılmayacaktır.
