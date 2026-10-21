# Calibration-set v1 insan inceleme paketi

Durum: **Paket üretildi ve teknik olarak doğrulandı; insan etiketlemesi başlamadı.**

Tarih: 2026-10-21

## Amaç

Bu paket, `calibration-set v1` içindeki 278 belirsiz adayı aynı biçimde ve izlenebilir Pair ID ile
insan gözüyle değerlendirmek için hazırlanmıştır. Görsel çift incelemesi kanser tanısı koymaz;
yalnız iki dosyanın aynı kaynak görüntünün dönüşümleri olup olmadığına bakar.

Bu işlem:

- otomatik bir duplicate kararı vermez;
- pHash eşiği seçmez;
- 70/15/15 sınıflandırma split'i oluşturmaz;
- model eğitmez.

## Üretim komutu

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\render_calibration_review.py --pairs-per-page 6 --thumb-size 320
```

Kod mevcut `review/` klasörünü ezmez. Yerel çıktılar:

- `artifacts/audit/duplicate_calibration_v1/review/review_labels.csv`
- `artifacts/audit/duplicate_calibration_v1/review/contact_sheet_index.csv`
- `artifacts/audit/duplicate_calibration_v1/review/review_render_report.json`
- `artifacts/audit/duplicate_calibration_v1/review/contact_sheets/`

## Doğrulanmış sonuç

| Kontrol | Sonuç |
|---|---:|
| Kaynak review adayı | 278 |
| Development adayı | 139 |
| Holdout adayı | 139 |
| Benzersiz Pair ID | 278 |
| Contact-sheet indeks satırı | 278 |
| Temas sayfası | 64 |
| Eksik indekslenmiş sayfa | 0 |
| Pair ID/path eşleme hatası | 0 |
| Boş `human_verdict` | 278 |
| `pending_visual_review` | 278 |
| Temas sayfası boyutu | Yaklaşık 82,34 MiB |

Çift türleri:

- 104 `random_same_leaf_review`;
- 104 `phash_hard_same_leaf_review`;
- 70 `cross_leaf_same_source_group_review`.

## Sayfada görülen bilgiler

Her çift için:

- kalıcı `C000...` Pair ID;
- `development` veya `holdout` audit bölümü;
- adayın üretim türü;
- source group;
- sol ve sağ alt sınıf;
- iki dosya adı;
- pHash Hamming mesafesi;
- yan yana 320×320 görüntü kutuları

gösterilir. pHash mesafesi yalnız destekleyici kanıttır; insan kararını otomatik belirlemez.

## İnsan karar sözlüğü

| Karar | Ne zaman kullanılır? |
|---|---|
| `same_source_candidate` | Aynı anatomik/hücresel düzenin dönüşmüş, kaymış, dönmüş, aynalanmış, kırpılmış veya parlaklığı değişmiş hâli olduğuna güçlü kanıt varsa |
| `independent_lookalike` | Aynı doku/sınıf/modaliteye benzese bile yerel yapıların farklı bağımsız görüntüler olduğunu gösteren kanıt varsa |
| `uncertain` | Görüntü çözünürlüğü veya benzerlik güvenli karar vermeye yetmiyorsa |

Her karara `high`, `medium` veya `low` confidence ve gerekirse kısa not eklenir. Boş karar hiçbir
zaman otomatik negatif sayılmaz.

## Development/holdout kontrolü

İnsan incelemesi iki aşamalı yürütülecek:

1. Önce yalnız 139 **development** adayı etiketlenecek.
2. Hibrit yöntemin kuralları/eşikleri yalnız development verisiyle ayarlanıp dondurulacak.
3. 139 **holdout** adayı bu sırada görülmeden ve kullanılmadan kapalı kalacak.
4. Yöntem dondurulduktan sonra holdout insan referansı tamamlanıp yöntem yalnız bir kez ölçülecek.

Bu ayrım, eşiği sınav cevaplarına bakarak ayarlamamızı önler. Audit development/holdout ayrımı,
gelecekteki kanser sınıflandırıcısının train/validation/test split'i değildir.

## Görsel kalite kontrolü

Yalnız üç development sayfası açılarak düzen kontrol edildi:

- çapraz-alt-sınıf brain MRI;
- pHash-hard lung/colon histoloji;
- rastgele Pap smear.

Pair ID, sınıf, dosya adı ve pHash metinleri okunabilir; sol/sağ görüntüler doğru yerleşmiştir.
Holdout sayfaları açılmadı ve hiçbir çift için insan kararı kaydedilmedi.

## Bir sonraki durma noktası

Sonraki iş, development çiftlerini küçük ve izlenebilir partiler hâlinde birlikte etiketlemektir.
İlk önerilen parti bir source group içindeki tek sayfadır. Her karar Pair ID üzerinden
`review_labels.csv` dosyasına işlenecek; sayfa tamamlanmadan sonraki gruba geçilmeyecektir.
