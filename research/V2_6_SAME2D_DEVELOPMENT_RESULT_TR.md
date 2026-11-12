# V2.6 aynı-2D leakage-family geliştirme sonucu

Tarih: 2026-11-12

## Neyi dondurduk?

Mehmet'in onayıyla projenin savunabileceği leakage-family hedefi
`same_2d_augmentation_family_v1` olarak donduruldu. Aynı kaynak 2D görüntünün
döndürme, çevirme, parlaklık/kontrast değişimi, yeniden boyutlandırma veya kırpma
gibi türevleri aynı ailedir. Farklı MRI/CT kesitleri, aynı organın benzer görüntüleri
ve yalnız yüksek embedding benzerliği aynı aile için yeterli değildir.

Veri setinde hasta, tarama ve seri kimliği bulunmadığı için bu hedef hasta-düzeyi
ayrımı kanıtlamaz. Rapor açıkça yalnız exact ve aynı-2D augmentation-family
sızıntısını kontrol ettiğimizi söylemelidir.

Hedef dosyası: `config/leakage_family_target_v1.json`

SHA-256: `25c46dec6a912896ad22a6832dbc1ffee40e37650766d970453b4a110840d150`

## Ne çalıştırıldı?

V2.5 prospective audit'te görsel olarak “same_source_candidate” denilen 16 outside
çift, dondurulmuş duplicate scorer ve `duplicate_rule_v1` ile yerelde yeniden
ölçüldü. Bu işlem 12,69 saniye sürdü; çift başına ortalama süre 0,788 saniyeydi.

İlk otomatik karar:

- 4 `same_source`
- 7 `independent`
- 5 `uncertain`

Beş belirsiz Kidney CT çifti orijinal çözünürlükte yeniden açıldı. Organ konturları,
renal kontrast dağılımları, bağırsak/çevre yapıları ve kesit seviyeleri birebir
örtüşmediği için aynı 2D görüntünün dönüşümü olmadıkları sonucuna varıldı. Bu beş
karar, eski V2.5 kararını yeniden yazmadan ayrı bir adjudication katmanında
`independent` olarak kaydedildi.

## Nihai geliştirme referansı

| Sonuç | Çift sayısı |
|---|---:|
| Aynı 2D kaynak | 4 |
| Bağımsız 2D görüntü | 12 |
| Belirsiz | 0 |

Aynı-2D çiftler:

- `V25N000103` — Kidney CT — minimum robust K=20
- `V25N000142` — Lymphoma histology — minimum robust K=24
- `V25N000145` — Lymphoma histology — minimum robust K=22
- `V25N000254` — Lymphoma histology — minimum robust K=29

Önemli sonuç: daha önce K=2.076 gerektiren `V25N000107` farklı bir 2D MRI
kesitidir; onaylanan hedefte aynı augmentation ailesi değildir. Bu nedenle K=2.076
artık candidate kuralı için bir recall zorunluluğu değildir.

Çıktılar:

- `research/v2_6_same2d_development_v1/pair_scores_and_rule.csv`
- `research/v2_6_same2d_development_v1/report.json`
- `research/v2_6_same2d_development_v1/final_pair_references.csv`
- `research/v2_6_same2d_development_v1/adjudication_report.json`

Final CSV SHA-256: `e5b93fd2eafed45ccdb535ecd2be0c87f6f4e1893666053fea6a60e933579bca`

## Candidate kuralı açısından anlamı

Dört gerçek aynı-2D kaçağın tümü aynı 26-sınıf yaprağındadır ve robust K=30 içinde
bulunur. Fakat Kidney ve Lymphoma gruplarının tamamını K=30 yapmak yaklaşık
3.469.520 çift, yani 26,689 çift/görüntü üretir ve dondurulmuş 25 kapasite sınırını
geçer.

Bu nedenle sıradaki dar karşı-olgusal şudur:

1. V2.5 adaylarının tamamını koru.
2. Brain MRI'yı genişletme.
3. Yalnız Kidney CT ve Lymphoma histology için aynı alt sınıftaki robust rank
   16–30 çiftlerini ekle.
4. Önce tam artifact üzerinde aday yükünü ve 4/4 geliştirme kapsamını ölç.
5. Geçerse yeni, path-disjoint ve score-blind prospective audit yapmadan kuralı
   kabul etme.

Bu belge model başarısı değildir; family, split veya eğitim sonucu içermez.
