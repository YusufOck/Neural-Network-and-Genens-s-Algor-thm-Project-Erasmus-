# Candidate Generator v2.4 — N000123 Full-Index Rank-Crowding Sonucu

Tarih: 2026-11-09  
İşlem türü: Önceden tanımlanmış, salt-okunur tam-index tanısı  
Durum: Tamamlandı; v2.5 kuralı değildir

## Amaç

V2.4 tam-artifact değerlendirmesinde kaçan doğal aynı-kaynak referansı `N000123` için şu iki
soru ölçüldü:

1. Çift, 25.000 görüntülük `cervix_pap_smear` indeksinde dört frozen descriptor kanalında
   gerçekte kaçıncı sıradadır?
2. Robust top-K değeri önceden tanımlanmış `15,20,25,30,40,50,75,100` noktalarına çıkarılırsa
   tam 130.000 görüntü için aday yükü 25 çift/görüntü kapasite sınırını geçer mi?

Bu çalışma duplicate kararı vermedi, family/split oluşturmadı, model eğitmedi ve v2.5 seçmedi.

## Çalıştırma ve bütünlük

- Colab commit'i: `068de872daa16e092de94ce2cf7f34510e7fb825`
- Cihaz: Tesla T4 / CUDA
- Run contract: `6c4884d25da3f5d2af2b6d64522b3a78af08db5e05e9ce99e4b13d12686a5261`
- Kayıtlı full-retrieval artifact: 1.107
- Bu tanıda hash ile doğrulanan artifact: 206
  - 1 image index
  - 5 serviks leaf descriptor artifact'ı
  - 4 serviks robust-retrieval artifact'ı
  - 196 serviks pHash block artifact'ı
- Config SHA-256: `c5073b99fcafcbfc87e691a8250d818ffc39ac2f7de8c8635e3c9c7bba867ab7`
- Analyzer SHA-256: `bcdafcb1c3a1df1d6b464c1fa4578ab3de1237ca6bb6eb06470e5008edab4484`
- Git'te saklanan rapor: `research/candidate_generator_v2_4_rank_crowding_v1.json`
- Git'te saklanan rapor SHA-256: `4e71e5a0a08fd75c7e029566a7a7571876818dc455b2f9f5bc867403fbed5a89`
- Bağımsız yapısal, hash ve aritmetik doğrulama: 42/42 geçti.

## Hücre çıktılarının yorumu

### Ortam ve ön kontrol

Drive bağlandı, doğru Git commit'i kullanıldı, CUDA ve Tesla T4 görüldü. V2.3 output klasörü,
`stage_manifest.json`, `run_contract.json`, frozen config ve analyzer bulundu. Stage toplamı
`26 + 1 + 26 + 1.022 + 32 = 1.107` olarak doğrulandı. ZIP veya dataset klasörü gerekmedi;
tanı yalnız önceden üretilmiş descriptor ve candidate artifact'larını okudu.

### Dört descriptor kanalındaki gerçek rank

| Kanal | Cosine similarity | Sol→sağ rank | Sağ→sol rank | En iyi yön |
|---|---:|---:|---:|---:|
| Raw full + flip mean | 0,863427 | 532 | 377 | 377 |
| Autocontrast full + flip mean | 0,887870 | 152 | 66 | 66 |
| Raw multicrop + flip mean | 0,905170 | 63 | 206 | 63 |
| Autocontrast multicrop + flip mean | 0,919602 | 34 | 56 | **34** |

Rank'ın iki yönde farklı olması normaldir: aynı cosine değerine rağmen her görüntünün çevresindeki
benzer komşu yoğunluğu farklıdır. Candidate union, çift en az bir yönde top-K içine girdiğinde
çifti kapsadığı için ilgili minimum eşik `34`'tür. Küçük 6.604-görüntülük audit indeksindeki rank
1'in full 25.000-görüntülük serviks indeksinde 34'e düşmesi, doğrudan full-scale rank crowding
kanıtıdır.

`N000123` için pHash mesafesi 34'tür ve full pHash≤12 kanalında yoktur. Dolayısıyla çifti geri
getiren kanal pHash değil, autocontrast + multicrop robust descriptor kanalıdır.

## Önceden tanımlanmış yük projeksiyonları

| Robust top-K | N000123 bulundu mu? | Tam aday | Çift/görüntü | ≤25 kapasite |
|---:|:---:|---:|---:|:---:|
| 15 | Hayır | 2.491.553 | 19,166 | Evet |
| 20 | Hayır | 2.567.048 | 19,747 | Evet |
| 25 | Hayır | 2.641.985 | 20,323 | Evet |
| 30 | Hayır | 2.729.755 | 20,998 | Evet |
| 40 | **Evet** | 2.994.870 | 23,037 | **Evet** |
| 50 | Evet | 3.388.432 | 26,065 | Hayır |
| 75 | Evet | 4.390.768 | 33,775 | Hayır |
| 100 | Evet | 5.340.424 | 41,080 | Hayır |

Gerçek minimum rank 34'tür; ancak 34 önceden tanımlanmış yük grid'inde olmadığı için onun tam aday
sayısı hesaplanmadı. Grid içinde çifti ilk kapsayan nokta K=40'tır. K=40, K=15'e göre 503.317
ek aday (%20,20 artış) üretir ve kapasite tavanının 255.130 aday altında kalır. K=50 ise kapasiteyi
138.432 aday aşar.

## Baseline yeniden üretimi

Tanı, yeni K noktalarını yorumlamadan önce frozen baseline'ı birebir yeniden üretti:

- Stored robust top-15: 355.588
- Recomputed robust top-15: 355.588
- Stored pHash: 68.770
- Serviks top-15 ∪ pHash: 378.508
- Full v2.4: 2.491.553

Bu eşleşme, yeni projeksiyonların aynı index, descriptor, tie-breaking ve union mantığından
hesaplandığını gösterir.

## Bilimsel karar

1. V2.4'ün `N000123` kaçağının nedeni MRI/CT pHash kapatma kararı değildir; serviks grubunda
   robust top-15'in full-index rank crowding altında yetersiz kalmasıdır.
2. Bu tek referans için gereken minimum robust top-K 34'tür.
3. Önceden tanımlanmış grid içinde K=40 hem referansı geri getirir hem mevcut 25 çift/görüntü
   kapasite sınırını geçmez.
4. Buna rağmen K=40 henüz v2.5 olarak kabul edilemez. Sonuç yalnız bir doğal pozitif üzerindeki
   odaklı tanıdır; başka doğal akrabaların top-40 dışında olmadığını kanıtlamaz ve adayların gerçek
   duplicate oranını ölçmez.
5. Family, split ve classifier eğitimi hâlâ başlamamalıdır. Sonraki adım, bu kanıtı kullanarak
   sonuç görülmeden tanımlanmış v2.5 politika/acceptance planını hazırlamak ve ayrıca onay almaktır.

