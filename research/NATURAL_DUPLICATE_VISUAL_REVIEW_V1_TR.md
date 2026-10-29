# Natural near-duplicate görsel inceleme v1

Tarih: 2026-10-29

Durum: **345/345 çift score-blind olarak incelendi ve ayrı karar kaynağına
donduruldu.** Bu çıktı bir kanser sınıflandırma sonucu, klinik uzman etiketi,
family listesi veya train/validation/test split'i değildir.

## Yöntem

- Girdi: `research/natural_duplicate_audit_v1/review_manifest.csv`
- Dondurulmuş girdi SHA-256:
  `84ae0e9fa6d09cf6162d5ecbeec47cab82786217a3895d03c67ed8e55729da7c`
- İnceleme yüzeyi: 58 temas sayfası, altı çift/sayfa; zor çiftlerde orijinal
  512×512 görüntüler ayrıca açıldı.
- İnceleme sırasında private seçim manifesti, stratum, pHash uzaklığı, robust
  rank/benzerlik ve frozen-rule skorları açılmadı.
- Reviewer: `Codex AI-assisted visual review authorized by Mehmet`
- Karar dosyası: `research/natural_duplicate_visual_review_v1.csv`
- Karar dosyası SHA-256:
  `91d857685ada7dbd6d3fb6e3add5a8fe914e3767389f1d37401a660ffd308bca`

## Kör görsel kararlar

| Karar | Sayı |
|---|---:|
| `same_source_candidate` | 21 |
| `independent_lookalike` | 322 |
| `uncertain` | 2 |
| Toplam | 345 |

`same_source_candidate` Pair ID'leri:

`N000021`, `N000055`, `N000064`, `N000075`, `N000098`, `N000120`,
`N000136`, `N000145`, `N000162`, `N000170`, `N000220`, `N000234`,
`N000239`, `N000248`, `N000289`, `N000294`, `N000304`, `N000308`,
`N000333`, `N000342`, `N000345`.

İki abstention:

- `N000252`: CT anatomisi ve tümör görünümü yakın; bunun komşu acquisition
  slice'ı mı yoksa bağımsız benzer örnek mi olduğu yalnız görüntüden güvenli
  biçimde belirlenemedi.
- `N000337`: Yoğun Pap-smear dokusu çok benzer, fakat birebir tekrar eden
  landmark eşlemesi zorla karar vermeye yetecek kadar güvenilir değildi.

## Yorum sınırı

Bu kararlar AI-destekli, uzman-olmayan bir leakage-audit referansıdır; patoloji
veya radyoloji gold standard'ı değildir. `same_source_candidate` ifadesi de
otomatik olarak nihai family bağı kurmaz. İki `uncertain` çift sonraki scorer
karşılaştırmasında zorla pozitif veya negatif yapılmayacaktır.

Skor/stratum bilgileri ancak bu dosya Git ile dondurulduktan sonra açılacak.
Sonraki adım, kör kararları Candidate Generator v2.3 strata ve frozen duplicate
rule v1 çıktısıyla karşılaştırmaktır. Özellikle `outside_*` strata içinde bir
`same_source_candidate` bulunması full-scan kapısını engelleyecektir.

