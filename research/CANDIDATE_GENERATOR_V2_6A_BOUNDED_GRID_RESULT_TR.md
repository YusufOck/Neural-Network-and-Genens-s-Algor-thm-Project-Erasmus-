# Candidate Generator v2.6a Bounded Grid Sonucu

Tarih: 2026-11-13
Durum: full-artifact development tanısı geçti; selected profile prospective doğrulama bekliyor

## Doğrulanan sonuç

- Rapor SHA-256: `1b678e715e509e038193175b9c363c3ad20b32ba1ab3e10617e0ec5887591bf7`.
- 132 ilgili Drive artifact'ı hash ile doğrulandı.
- Frozen four-channel top-15 tabanı birebir yeniden üretildi.
- Üç profil de dört same-2D geliştirme referansını 4/4 buldu.
- All-channel profil 3.265.225 aday = 25,117115 çift/görüntü ile kapasiteyi geçemedi.
- Kidney-all/Lymphoma-raw-multicrop profil 3.116.598 aday = 23,973831 çift/görüntü ile geçti.
- Raw-multicrop-only profil 3.099.037 aday = 23,838746 çift/görüntü ile geçti.

Önceden dondurulan `maximum_candidate_coverage_under_capacity_then_config_order` kuralı
`kidney_all_lymphoma_raw_multicrop` profilini seçti. Seçilen profil kapasite altında 133.402
çiftlik boşluk bırakır. Bu development sonucu aday kuralını kabul ettirmez; yalnız yeni,
path-disjoint prospective audit hipotezidir.

## Seçilen uzatma

- Complete v2.5 union korunur.
- Kidney CT: aynı-leaf rank 16–20, dört descriptor kanalının birleşimi.
- Lymphoma: aynı-leaf rank 16–29, yalnız `raw_multicrop_flip_mean`.
- Cross-leaf top-15, v2.5 pHash politikası ve exact SHA-256 union korunur.

Scoring, family, split ve classifier training başlamadı.
