# Candidate Generator V2.6 aynı-yaprak karşı-olgusal planı

Tarih: 2026-11-12

## Amaç

Split'e giden yolu hızlandırmak için V2.5'in tüm güvenli adaylarını koruyan ve
yalnız doğrulanmış aynı-2D geliştirme kaçaklarının bulunduğu dar alanı genişleten
bir candidate hipotezinin tam veri yükünü ölçmek.

## Sonuç görülmeden dondurulan kural

- Başlangıç: V2.5 full union, 2.994.870 çift.
- Mevcut robust top-15 ve pHash politikası aynen korunur.
- Cross-leaf robust top-15 adayları korunur.
- Kidney CT ve Lymphoma histology içinde yalnız aynı 26-sınıf leaf'e ait rank
  16–30 robust çiftleri eklenir.
- Brain MRI için ek genişletme yapılmaz.
- Exact duplicate union zorunlu kalır.

Config:
`config/candidate_generator_v2_6_same_leaf_counterfactual.json`

Config SHA-256:
`75fad585725d9f3a88d2dd3ee1bcff00433c53fc26c8be6c8be330cb1ec3ef6d`

## Bu koşudaki kapılar

1. Kidney ve Lymphoma frozen top-15 birleşimleri birebir yeniden üretilmeli.
2. Dört geliştirme aynı-2D referansının 4/4'ü proposal içinde olmalı.
3. Toplam yük 25 çift/görüntüyü geçmemeli.

Bu üç kapının geçmesi V2.6'yı kabul ettirmez. Yalnız yeni prospective doğrulamaya
geçmeye izin verir.

## Geçerse hızlandırılmış yol

1. Küçük fakat path-disjoint score-blind audit seçimi.
2. Outside bölgesinde kesin aynı-2D çift ve çözümsüz belirsiz bulunmaması.
3. Kuralı dondurma ve tam candidate dosyasını üretme.
4. Frozen scorer ile candidate kararlarını üretme.
5. Exact + onaylı near-duplicate bağlantılarından family component'leri kurma.
6. 26 alt sınıfı koruyan, aileleri bölmeyen sabit 70/15/15 manifesti oluşturma.
7. Aile kesişimi ve sınıf dağılımı testlerinden sonra model eğitimine geçme.

## Yasak yorumlar

- Dört geliştirme çiftini bulmak prospective recall kanıtı değildir.
- Candidate olmak duplicate olmak değildir.
- Aynı leaf'te olmak aynı aile olmak değildir; yalnız aday aramasını daraltır.
- Hasta/seri kimliği olmadığı için hasta-düzeyi leakage kontrolü iddia edilemez.
- Bu koşu split veya model eğitimi başlatmaz.
