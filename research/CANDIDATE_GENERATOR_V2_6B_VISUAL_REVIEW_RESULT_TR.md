# Candidate Generator v2.6b — Kör Görsel İnceleme Sonucu

Tarih: 2026-11-14  
Deney türü: path-disjoint, score-blind prospective görsel denetim  
Hedef: `same_2d_augmentation_family_v1`

## Bu adımda ne yapıldı?

Private seçim tablosu açılmadan, public manifestteki 160 çiftin tamamı görsel olarak
incelendi. İlk geçişte 27 temas sayfası, sınırdaki çiftlerde ise yalnız public
görüntülerden üretilen tek-çift/yüksek-çözünürlüklü sayfalar kullanıldı.

Pozitif karar için yalnız genel doku benzerliği yeterli sayılmadı. Aynı 2B görüntünün
flip, döndürme, kaydırma, crop veya görünüm değişikliği sonrasında korunması beklenen
birden fazla ince anatomik/doku işaretinin birebir uyuşması arandı. Farklı CT/MRI
kesitleri, aynı organa benzese bile bu hedefte bağımsız kabul edildi.

## Dondurulan kararlar

| Görsel karar | Çift sayısı |
|---|---:|
| `same_source_candidate` | 24 |
| `independent_lookalike` | 135 |
| `uncertain` | 1 |
| **Toplam** | **160** |

Güven dağılımı: 126 yüksek, 33 orta, 1 düşük. Pair ID sayısı 160/160 benzersizdir;
boş karar, boş reviewer veya eksik satır yoktur.

## Bütünlük ve körlük kanıtı

- Public giriş manifesti SHA-256:
  `f961d5e92eb4da2b581852e8151834ff16deaa245a3d7e6e35fb819acd4fb099`
- Dondurulan karar CSV SHA-256:
  `2220fca301f0d281cab03f98847e865e02de1269b0881479ebae7813d7b4f36b`
- Karar dondurma kodu SHA-256:
  `7da29f9f5b1f6691616a31b622d77b3fdc459467889a1904d5ac54d5cae04f8c`
- Private CSV'nin ham SHA-256'sı inceleme öncesi değerle aynıdır:
  `549e12b57daac220fbfd631f2717c28f5b78370f42fb713afdb746e176a7f955`
- Private CSV satırları okunmadı veya parse edilmedi.

Karar dosyası:
`research/candidate_generator_v2_6b_visual_review_v1.csv`

## Bu sonuç ne anlama gelmez?

Ham olarak 24 pozitif bulunması v2.6b kuralının geçtiğini veya kaldığını göstermez.
Bir pozitif `candidate_new_extension` içindeyse beklenen retrieval kanıtıdır; 128
outside challenge satırından birindeyse önceden dondurulan sıfır-kaçak kapısını bozar.
Hangi satırın hangi grupta olduğu private CSV açılmadan bilinemez.

Bu nedenle bu aşamada:

- candidate kuralı kabul edilmedi,
- 3.116.598 aday skorlanmadı,
- duplicate family oluşturulmadı,
- train/validation/test split yapılmadı,
- model eğitimi başlatılmadı.

## Sonraki kontrollü adım

Mehmet'in ayrı onayından sonra private tablo bir kez açılarak 160 dondurulmuş karar
Pair ID üzerinden birleştirilecek ve önceden tanımlı prospective kapılar hesaplanacak.
Bu tek seferlik unblinding tamamlanmadan family/split/eğitime geçilmeyecektir.
