# Candidate Generator v2.5 — Prospective Audit Freeze Kaydı

Tarih: 2026-11-09  
İşlem türü: Kod/config hazırlığı ve sonuç öncesi hash freeze  
Durum: Hazır; full selection henüz çalıştırılmadı ve hiçbir görüntü açılmadı

## Dondurulan aday politikası

- `cervix_pap_smear`: dört-kanal robust top-40
- Diğer yedi source group: dört-kanal robust top-15
- pHash≤12: ALL, breast, cervix, lung/colon, lymphoma ve oral gruplarında açık
- pHash: brain MRI ve kidney CT gruplarında kapalı
- Exact SHA-256 çiftleri: zorunlu global union
- Cross-source-group candidate: yasak

Bu politika `N000123` sonucundan sonra tasarlandığı için eski 23 doğal pozitif bağımsız kabul
kanıtı değil, regression kontrolüdür. Yeni score-blind outside katmanları prospective sınavdır.

## Dondurulan dosyalar

| Dosya | SHA-256 |
|---|---|
| `config/candidate_generator_v2_5_prospective_audit.json` | `2c4213d34c70991550b85ec2d4699a0c7cf49b804904cc9bb3b4d7518d8fe745` |
| `src/build_candidate_generator_v2_5_prospective_audit.py` | `865342f0eab1730b7bd0042b4bc9946784f3e1bfc75f591fcf2a72108c160c59` |
| `src/build_candidate_generator_v2_5_exclusion_manifest.py` | `2ada97f2352e52e9126da96d710b0f2eaf9d2697cfd3986e7a2e137f59cc840a` |
| `research/candidate_generator_v2_5_excluded_paths_v1.json` | `bc90f337d0058648c27312886cd407177a7e329a85373b13c58f4ab68bf38c3f` |

Config kendi `.sha256` dosyasıyla ve builder hash'i config içinden doğrulanır. Builder ayrıca
tracked v2.4 counterfactual ve rank-crowding raporlarının hash'lerini doğrulamadan seçim yapmaz.

## Exclusion manifesti

Metadata-only generator şu dondurulmuş yerel kanıtları birleştirdi:

| Kaynak | Yol |
|---|---:|
| V2.1 holdout dataset görüntüleri | 6.604 |
| V2.2 holdout / önceki natural-audit havuzu | 6.604 |
| Scaled candidate pilot dataset görüntüleri | 7.208 |
| Kalibrasyon dataset görüntüleri | 708 |
| Exact-duplicate grup üyeleri | 2.673 |
| **Benzersiz birleşim** | **22.922** |

Bazı kaynaklar birbiriyle kesiştiği için birleşim satır toplamından küçüktür. Canonical path-list
SHA-256: `1569fa19392cb840b81bfc018e5849050774f3f36beb5a5478281b9d2e0be0f6`.
`N000123`'ün iki yolu bu manifestte bulunur. Full 130.000-image index içinde beklenen eligible
görüntü sayısı 107.078'dir; bu sayı full selection sırasında yeniden doğrulanacaktır.

## Builder güvenlikleri

Builder:

1. 1.107 artifact stage envanterini, run contract'ı, dosya boyutlarını ve SHA-256 hash'lerini
   doğrular.
2. Dört descriptor kanalında full-index stable rank'ları yeniden üretir.
3. Her grubun recomputed robust top-15 union'ını frozen v2.3 artifact'ıyla birebir karşılaştırır.
4. Cervix K=15 ve K=40 union sayılarını 378.508 / 881.825 olarak doğrular.
5. Full v2.5 sayısı tam 2.994.870 ve yük ≤25 değilse durur.
6. Exclusion manifestindeki hiçbir görüntüyü seçmez.
7. Her yeni görüntüyü en fazla bir çiftte kullanır.
8. Önce outside challenge katmanlarını seçerek endpoint yarışında kabul kanıtına öncelik verir.
9. Private CSV'de skor/stratum tutar; review CSV'de bunları gizler ve tüm kararları boş bırakır.
10. Mevcut v2.3 artifact klasörünün içine çıktı yazmayı reddeder.

## Test sonucu

- Yeni v2.5 testleri: 10/10 geçti.
- Bütün proje testleri: 127/127 geçti.
- Python syntax ve `git diff --check`: geçti.

Bu testler kod/config bütünlüğünü kanıtlar; henüz 256 çiftin seçildiğini veya audit'in geçtiğini
göstermez.

## Sonraki kontrollü adım

Yeni bir Colab/T4 çalıştırıcısı hazırlanıp yalnız full selection çalıştırılacaktır. Beklenen çıktı:

- `selection_private.csv`
- `review_manifest.csv` — tüm karar alanları boş ve skorlar gizli
- `selection_report.json`
- Drive konsol logu

Bu üç seçim artifact'ı hash ile doğrulanıp GitHub'a dondurulmadan hiçbir seçili görüntü
açılmayacak veya render edilmeyecektir. Colab selection çalışması için ayrıca Mehmet onayı gerekir.

