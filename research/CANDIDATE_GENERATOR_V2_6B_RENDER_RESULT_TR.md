# Candidate Generator v2.6b — Score-Blind Render Sonucu

Tarih: 2026-11-14

İşlem türü: Prospective görsel-audit altyapısı / public-only render

Durum: Temas sayfaları üretildi ve doğrulandı; görsel kararlar henüz başlamadı

## Amaç

Dondurulmuş 160 prospective çift, private seçim bilgileri açılmadan insan gözüyle
karşılaştırılabilir temas sayfalarına dönüştürüldü. Bu işlem duplicate kararı vermez ve
v2.6b candidate kuralını kabul etmez.

Renderer yalnız `selection_report.json` ile public `review_manifest.csv` dosyasını okur.
`selection_private.csv` okunmaz. Sayfalarda yalnız Pair ID, source group, sol/sağ işareti ve
iki görüntü gösterilir; candidate/outside üyeliği, stratum, rank, descriptor kanalı ve pHash
bilgileri gizli kalır.

## Doğrulanmış çıktı

| Kontrol | Sonuç |
|---|---:|
| Public çift | 160 |
| Benzersiz görüntü | 320 |
| Sayfa başına çift | 6 |
| Temas sayfası | 27 |
| Index satırı / benzersiz Pair ID | 160 / 160 |
| Boş review kararı | 160 / 160 |
| Render bütünlük kontrolü | 12 / 12 geçti |
| Tam proje test paketi | 165 / 165 geçti |

İlk ve son sayfa yalnız layout QA amacıyla açıldı. Başlık, sayfa numarası, `SCORE-BLIND`
işareti, Pair ID, source group ve L/R göstergeleri okunaklıydı. Kırpılma veya private bilgi
sızıntısı görülmedi. Hiçbir çifte görsel karar atanmadı.

## Bütünlük zinciri

| Artifact | SHA-256 |
|---|---|
| Renderer | `588c1acea9f2836a5c82fac00e21a053c2544e9066fb1f80dfbd9f53be94642e` |
| `render_report.json` | `b0c749b0952bf8f868ac073e345b914f35443a21a344c42fd6b6ef8193653398` |
| `sheet_index.csv` | `10017781293121f525a180af95d1ce9672a95ad7121eb2243b3fa8310180e288` |
| Frozen public review manifest | `f961d5e92eb4da2b581852e8151834ff16deaa245a3d7e6e35fb819acd4fb099` |
| Unchanged private manifest | `549e12b57daac220fbfd631f2717c28f5b78370f42fb713afdb746e176a7f955` |

PNG dosyaları büyük ve yeniden üretilebilir oldukları için Git'e eklenmez; yerel konumları:

`artifacts/audit/candidate_generator_v2_6b_prospective_review_v1/contact_sheets/`

Küçük metadata kopyaları:

`research/candidate_generator_v2_6b_prospective_review_v1/`

## Bilimsel anlamı ve sonraki kapı

Bu sonuç yalnız kör görsel inceleme paketinin doğru ve izlenebilir hazırlandığını kanıtlar.
V2.6b recall kapısı hâlâ değerlendirilmedi; family, split ve model eğitimi başlamadı.

Sonraki ayrı onaylı adım, 27 sayfadaki 160 çifti private bilgi görmeden
`same_source_candidate`, `independent_lookalike` veya `uncertain` olarak incelemek ve bütün
public kararları hash ile dondurmaktır. Ancak bundan sonra private tablo bir kez açılarak 128
outside satırda doğrulanmış same-2D kaçak bulunup bulunmadığı ölçülebilir.
