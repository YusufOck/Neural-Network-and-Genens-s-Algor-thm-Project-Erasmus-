# Candidate Generator v2.6a Bounded Grid Planı

Tarih: 2026-11-12
Durum: config/kod hash-kilitli; full-artifact Colab koşusu başlamadı

## Amaç

V2.6 dört same-2D geliştirme çiftini buldu fakat 25 çift/görüntü sınırını aştı. Bu tanı,
V2.5 birleşimini eksiltmeden üç dar profil karşılaştırır:

1. Kidney K=20 ve Lymphoma K=29; iki grupta dört descriptor kanalının tümü.
2. Kidney K=20 dört kanal; Lymphoma K=29 yalnız `raw_multicrop_flip_mean`.
3. Kidney K=20 ve Lymphoma K=29; iki grupta yalnız `raw_multicrop_flip_mean`.

Her uzatma yalnız aynı leaf içindeki rank 16–K çiftlerine uygulanır. Cross-leaf top-15,
v2.5'in geri kalanı, pHash politikası ve exact-duplicate birleşimi korunur.

## Önceden dondurulmuş seçim kuralı

- Dört geliştirme same-2D çiftinin 4/4'ü bulunmalıdır.
- Full yük en fazla 3.250.000 çift, yani 25 çift/görüntü olmalıdır.
- İki kapıyı geçen profiller arasında aday sayısı en yüksek olan profil seçilir; eşitlikte
  config sırası kullanılır. Amaç, kapasite içinde mümkün olan en geniş recall kapsamını
  korumaktır.
- Seçim yalnız prospective audit hipotezidir; kabul sonucu değildir.

## Güvenlik sınırları

- 1.107 v2.3 artifact salt-okunur kalır.
- Dataset ZIP'i/görüntü klasörü gerekmez.
- Duplicate scoring, family, split ve classifier training yapılmaz.
- Hasta/series düzeyi leakage kontrolü iddia edilmez.

## Beklenen çıktı

`MyDrive/NNGA_OUTPUTS/candidate_generator_v2_6a_bounded_grid_v1.json`

Rapor üç profil için yeni/toplam aday yükünü, 4/4 kapsamı, kapasite kapısını ve önceden
dondurulmuş politikaya göre seçilen geliştirme profilini gösterecektir.
