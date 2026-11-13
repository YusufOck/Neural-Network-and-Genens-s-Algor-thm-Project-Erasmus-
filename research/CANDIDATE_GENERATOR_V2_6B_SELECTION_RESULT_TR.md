# Candidate Generator v2.6b — Prospective Selection Sonucu

Tarih: 2026-11-13

İşlem türü: Full-index, path-disjoint, score-blind pair selection

Durum: Selection tamamlandı ve artifact'lar donduruldu; görsel audit henüz başlamadı

## 1. Çalıştırma ortamı

- Colab commit: `4a18ea42c15120eb40806226ebe1f65aad75831d`
- Cihaz: Tesla T4 / CUDA
- Frozen config ve builder hash kontrolü: geçti
- V2.3 kayıtlı artifact: 1.107
- Bu selection tarafından kullanılan ve hash ile doğrulanan artifact: 137
- Return code: 0
- Dataset ZIP'i veya görüntü klasörü bu selection için okunmadı

## 2. Paket ve artifact bütünlüğü

Colab paketi SHA-256:

`353d4258ca4ae9a225a2dbba066458127db708c4ba564ed8649cda274006bdb5`

Git'e alınan artifact'lar:

| Dosya | SHA-256 |
|---|---|
| `selection_report.json` | `3d00f465a7fbdf89065bf47f373d87d7d14fb9a10710e36a37dc171979a031cc` |
| `selection_private.csv` | `549e12b57daac220fbfd631f2717c28f5b78370f42fb713afdb746e176a7f955` |
| `review_manifest.csv` | `f961d5e92eb4da2b581852e8151834ff16deaa245a3d7e6e35fb819acd4fb099` |

`selection_private.csv` içeriği açılmadı, decode edilmedi veya satır olarak parse edilmedi. Yalnız
ham dosya SHA-256 değeri rapordaki önceden yazılmış hash ile karşılaştırıldı. Public
`review_manifest.csv` private stratum, candidate üyeliği, rank, descriptor kanalı veya pHash
alanı taşımadığı için yapı ve boş-karar kontrollerinde okunabildi.

## 3. Candidate policy

V2.5 candidate tabanı korunur. Yeni seçilen profil:

`kidney_all_lymphoma_raw_multicrop`

- Kidney CT: aynı-leaf, dört descriptor kanalında rank 16–20
- Lymphoma histology: aynı-leaf, yalnız raw-multicrop kanalında rank 16–29
- Toplam candidate: 3.116.598
- Yük: 23,9738308 çift/görüntü
- Frozen kapasite sınırı: en fazla 25 çift/görüntü
- Kapasite kapısı: geçti

Grup düzeyinde yeni aynı-leaf çift sayıları:

| Source group | V2.5 union | Yeni | V2.6b candidate union |
|---|---:|---:|---:|
| Kidney CT | 115.349 | 36.825 | 152.174 |
| Lymphoma histology | 339.509 | 84.903 | 424.412 |

## 4. Exclusion ve kör seçim

- Full index: 130.000 görüntü
- Önceki development/audit exclusion: 22.922 yol
- Önceki V2.5 prospective review: 512 yeni yol
- Birleşik exclusion: 23.434 yol
- Eligible: 106.566 görüntü
- Seçilen çift: 160/160
- Benzersiz görüntü: 320/320
- Önceki exclusion ile kesişim: 0
- Yerel veri setinde bulunamayan seçili yol: 0
- Benzersiz Pair ID: 160/160
- Review kararları: 160/160 boş
- Public CSV'ye sızan private stratum/rank/channel/pHash alanı: 0

Public source-group dağılımı:

- Kidney CT: 64 çift
- Lymphoma histology: 96 çift

## 5. Stratum kapsamı

| Private stratum | Hedef | Gerçek | Eksik |
|---|---:|---:|---:|
| `outside_selected_boundary` | 64 | 64 | 0 |
| `outside_alternate_channel` | 32 | 32 | 0 |
| `outside_random_same_leaf` | 32 | 32 | 0 |
| `candidate_new_extension` | 32 | 32 | 0 |
| **Toplam** | **160** | **160** | **0** |

Bu tablo yalnız selection raporundaki toplu sayımları gösterir. Hangi Pair ID'nin hangi stratum'a
ait olduğu görsel kararlar tamamlanana kadar açılmayacaktır.

## 6. Bu sonuç neyi kanıtlar?

Bu sonuç şunları kanıtlar:

1. Frozen v2.6b selection kodu önceden seçilen candidate profilini yeniden üretti.
2. 160 çift önceki 23.434 görüntü yolundan tamamen ayrıdır.
3. Her görüntü yalnız bir çiftte kullanılmıştır.
4. Review manifesti kördür ve henüz hiçbir görsel karar içermez.
5. Candidate yükü önceden dondurulan kapasite sınırının altındadır.

Bu sonuç şunları **kanıtlamaz**:

- 128 outside çiftin içinde same-2D kopya bulunmadığını;
- v2.6b candidate kuralının prospective kabul kapısını geçtiğini;
- 32 candidate kontrolünün gerçekten same-2D olduğunu;
- family, train/validation/test split veya model eğitimine geçilebileceğini.

## 7. Sonraki kontrollü adım

Bu artifact'lar GitHub'da dondurulduktan sonra yalnız public `review_manifest.csv` kullanılarak
160 çift için Pair ID'li, score-blind temas sayfaları üretilecektir. Renderer private CSV'yi
okumamalıdır. Görsel kararlar public manifestte tamamlanıp ayrıca hash ile dondurulmadan private
selection tablosu açılmayacaktır.

Prospective kabul kuralı değişmez: 128 outside satırdan herhangi birinde doğrulanmış same-2D çift
bulunursa v2.6b reddedilir. Görsel audit tamamlanmadan family, split veya classifier training yoktur.
