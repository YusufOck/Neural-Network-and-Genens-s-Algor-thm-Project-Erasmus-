# Natural duplicate adjudication v1 inceleme paketi

Tarih: 2026-11-04

Durum: **Paket hazır; 25/25 son-karar alanı bilinçli olarak boştur.** Bu aşama
bir ikinci-aşama veri denetimi hazırlığıdır. Duplicate family, train/validation/test
split veya kanser sınıflandırma modeli üretilmemiştir.

## Neden bu paket var?

Natural audit v1 içindeki 345 çiftin 320'sinde score-blind görsel inceleme ile
dondurulmuş duplicate rule v1 güvenli biçimde aynı kararda birleşti:

- 21 çift: ikisi de `same_source` yönünde;
- 299 çift: ikisi de `independent` yönünde.

Kalan 25 çiftte kararlardan biri belirsizdir veya iki karar doğrudan çatışır.
Bu çiftleri otomatik olarak family bağlantısına dönüştürmek yerine ayrı ve
izlenebilir bir adjudication kuyruğuna almak daha güvenlidir.

## Paket içeriği

| İçerik | Konum | Sayı |
|---|---|---:|
| Boş son-karar alanlı kuyruk | `research/natural_duplicate_adjudication_v1/adjudication_queue.csv` | 25 satır |
| Paket bütünlük raporu | `research/natural_duplicate_adjudication_v1/package_report.json` | 1 rapor |
| Pair-ID karşılaştırma kartları | `artifacts/audit/natural_duplicate_adjudication_v1/pair_views/` | 25 PNG |

Her PNG doğrudan Pair ID ile adlandırılmıştır. Örneğin `N000123` çifti için
`N000123.png` açılır. Kartta sol/sağ görüntü, önceki görsel karar, frozen-rule
kararı, pHash/robust retrieval bilgisi, ORB kanıtı ve en iyi hizalama metrikleri
birlikte gösterilir.

## Kuyruk dağılımı

| Kuyruk türü | Sayı | Anlamı |
|---|---:|---|
| `direct_conflict` | 2 | Görsel inceleme `independent`, frozen rule `same_source` dedi. |
| `visual_uncertain` | 2 | İlk score-blind görsel inceleme kesin karar veremedi. |
| `rule_uncertain` | 21 | Görsel inceleme bağımsız dedi; kural güvenli biçimde çekimser kaldı. |
| **Toplam** | **25** | Son değerlendirme bekliyor. |

Modalite dağılımı: 18 CT, 4 MRI, 2 Pap smear ve 1 histopatoloji çifti.

## Son karar kodları

- `same_source`: Aynı 2B kaynak görüntünün döndürülmüş, parlaklığı değiştirilmiş,
  kırpılmış veya başka biçimde augment edilmiş hali.
- `independent`: Farklı 2B görüntü. Aynı hastaya ait olabilecek komşu CT
  kesitleri de bu leakage-family tanımında bağımsız görüntülerdir.
- `uncertain`: Güvenli karar için kanıt yetersizdir; otomatik family bağlantısı
  kurulmaz.

## Bütünlük kontrolleri

- 345 dondurulmuş skor satırından yalnız güvenli konsensüs dışındaki 25 çift seçildi.
- 25 Pair ID benzersizdir.
- 25 CSV satırının her biri aynı Pair ID'li bir PNG ile eşleşir.
- 25/25 `adjudication_verdict` alanı boştur.
- İlk score-blind görsel karar dosyası ve frozen duplicate rule değiştirilmedi.
- Python 3.10 altında bütün proje testleri geçti: 76/76.
- Dondurulmuş skor girdisinin SHA-256 değeri:
  `76d8e739a9a59b26e7ef8de44bdb534e5f8be06b90d748dbfb7774e8c62df3da`.
- Hazırlanan kuyruk CSV SHA-256 değeri:
  `113fe1ca33e5ab34863b9ae36b8ba52f2d75d7eb2a2e2633b21f6445e750ce2e`.

## Sıradaki kontrollü adım

Önce priority 1 çiftleri (`N000123`, `N000206`), sonra görsel belirsiz çiftler,
son olarak rule-uncertain çiftler kanıtlarıyla incelenecek. Sonuçlar ilk dondurulmuş
kararların üzerine yazılmadan ayrı adjudication alanlarına kaydedilecek. Bu 25 karar
tamamlanıp doğrulanmadan Candidate Generator v2.3 dondurulmayacak ve 130.000
görüntülük tarama başlatılmayacaktır.
