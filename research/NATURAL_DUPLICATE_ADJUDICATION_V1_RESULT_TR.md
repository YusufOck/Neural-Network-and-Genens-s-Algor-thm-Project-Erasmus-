# Natural duplicate adjudication v1 sonucu

Tarih: 2026-11-04

Durum: **25/25 konsensüs-dışı çift ikinci aşamada incelendi ve ayrı karar
kaynağıyla sonlandırıldı.** Bu inceleme Codex tarafından, Mehmet'in yetkilendirmesiyle,
görüntüler ve önceden hesaplanmış skorlar görünürken yapılmıştır. Bağımsız klinik
uzman gold standard değildir.

Bu çıktı aynı 2B kaynak görüntünün augment edilmiş/dönüştürülmüş kopyalarını
araştırır. Hasta kimliği, klinik tanı veya aynı taramaya ait komşu kesit ailesi
üretmez.

## 1. Adjudication kararı

| Karar | Çift sayısı |
|---|---:|
| `same_source` | 2 |
| `independent` | 23 |
| `uncertain` | 0 |
| **Toplam** | **25** |

İki yeni aynı-kaynak kararı:

- `N000006`: Histoloji bez konturları, hücre dizilimleri ve küçük doku
  ayrıntıları dönüşüm altında birebir eşleşir. Robust rank 1 ve 1.507 ORB
  inlier aynı 2B kaynağı güçlü biçimde destekler. İlk score-blind görsel
  incelemede bağımsız denmiş, frozen rule ise çekimser kalmıştır.
- `N000123`: Pap-smear hücre kümeleri ve özgün çekirdek yerleşimleri
  dönüşüm/kırpma altında birebir eşleşir. Frozen rule bu çifti
  `strong_intensity_structure` yoluyla aynı-kaynak olarak işaretlemiştir.

Önemli bağımsız kararlar:

- `N000206`: Global CT geometrisi benzer olsa da bağırsak içeriği ile organ
  kesitleri farklıdır. Frozen rule `same_source` demiştir; adjudication bunu
  benzer veya komşu fakat farklı bir 2B kesit olarak değerlendirmiştir.
- `N000252`: Aynı taramanın komşu kesiti olabilir; tümör konturu ve çevre
  anatomi değiştiği için aynı 2B kaynak görüntü sayılmamıştır.
- `N000337`: Hücre/çekirdek yerleşimleri birebir örtüşmez ve geometrik kapsama
  çok düşüktür; bağımsız görüntü olarak değerlendirilmiştir.

## 2. 345 çiftlik nihai referans

| Nihai referans | Sayı |
|---|---:|
| `same_source` | 23 |
| `independent` | 322 |
| `uncertain` | 0 |
| **Toplam** | **345** |

23 aynı-kaynak çiftin tamamı Candidate Generator v2.3 aday bölgesindedir.
Önceden belirlenmiş outside challenge katmanlarında aynı-kaynak kararı yine
0'dır. Bu, incelenen challenge set için v2.3 hipotezini destekler; 3.093.466
olası doğal çiftin tamamında %100 recall kanıtı değildir.

## 3. Frozen duplicate rule ile ikinci-aşama karşılaştırma

Adjudication sırasında kural skorları görüldüğü için aşağıdaki değerler bağımsız
holdout performansı değildir. Yalnızca son referansla tanısal karşılaştırmadır.

| Nihai referans | Kural doğru | Kural `uncertain` | Kural karşı sınıf |
|---|---:|---:|---:|
| 23 aynı-kaynak | 22 | 1 (`N000006`) | 0 |
| 322 bağımsız | 300 | 21 | 1 (`N000206`) |

- Kapsama: 323/345 = %93,62.
- Kesin karar verilen çiftlerde doğruluk: 322/323 = %99,69.
- `uncertain` kararları doğru saymadan toplam doğruluk: 322/345 = %93,33.

Bu oranlar kanser sınıflandırma başarısı değildir.

## 4. İzlenebilir çıktılar

- Ayrı karar kaynağı:
  `research/natural_duplicate_adjudication_v1/adjudication_decisions_v1.csv`
- Kararların kanıt sütunlarıyla birleştirilmiş hali:
  `research/natural_duplicate_adjudication_v1/adjudicated_pairs_v1.csv`
- 345 çiftlik final referans tablosu:
  `research/natural_duplicate_adjudication_v1/final_pair_references_v1.csv`
- Makine-okunur rapor:
  `research/natural_duplicate_adjudication_v1/adjudication_report_v1.json`

Dondurulmuş boş kuyruk değişmemiştir. SHA-256:
`113fe1ca33e5ab34863b9ae36b8ba52f2d75d7eb2a2e2633b21f6445e750ce2e`.

Karar kaynağı SHA-256:
`f61944040d5a1d145ad785bd99f4de3e33fd801e90f367566a352607f09b551b`.

Python 3.10 proje testleri: **81/81 geçti**.

## 5. Sınırlar

- Kararlar AI-desteklidir; patolog/radyolog tarafından doğrulanmış değildir.
- Score-aware adjudication, önceki score-blind doğal denetimin yerine geçmez;
  onun ardından gelen ayrı ikinci aşamadır.
- Yerel dosyalarda hasta/scan kimliği bulunmadığından patient-level leakage
  güvenilir biçimde çözülemez.
- Komşu CT/MRI kesitleri bu çalışmada farklı 2B kaynak görüntülerdir. Bu tercih
  açıkça raporlanmalıdır.
- Henüz 130.000 görüntülük tarama, duplicate family, split veya model yoktur.

## 6. Sıradaki kontrollü adım

Candidate Generator v2.3 (`robust top-15 ∪ flip-aware pHash≤12`) uygulanmadan
önce konfigürasyonu ve kaynak kod hash'leri dondurulacaktır. Ardından 130.000
görüntülük descriptor/retrieval işlemi için GPU/Colab, aday doğrulaması için
CPU/paralelleştirme kapasite planı hazırlanacaktır.
