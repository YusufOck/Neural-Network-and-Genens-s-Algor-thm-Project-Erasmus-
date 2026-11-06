# Candidate Generator v2.4 tam-artifact karşı-olgusal sonucu

Tarih: 2026-11-06

Deney türü: **Önceden dondurulmuş, salt-okunur, tam-kapsamlı ablation**

Durum: **Çalışma teknik olarak tamamlandı; v2.4 kabul kapısını geçmedi.**

## 1. Test edilen soru

V2.3'ün robust dört-kanal top-15 adayları bütün source group'larda korundu. Flip-aware
`pHash<=12` yalnız `brain_ce_mri` ve `kidney_ct` gruplarında kapatıldı; bilinen exact
SHA-256 çiftleri zorunlu olarak birleşime eklendi. Amaç, doğal ve kontrollü referansları
kaçırmadan aday yükünü 25 çift/görüntünün altına indirmekti.

Bu çalışma duplicate kararı, family oluşturma, train/validation/test split'i veya model
eğitimi değildir.

## 2. Kanıt ve bütünlük

- Colab/Drive'dan indirilen özgün rapor SHA-256:
  `01e0d5c7f9662212e36283be2ed51045057538f303bba46b2e4965d2fa980b25`
- Git'te izlenen semantik arşiv kopyası:
  `research/candidate_generator_v2_4_counterfactual_v1.json`
- Arşiv kopyası SHA-256: `4bd51e7714db113e54efce6b132bf228e91ff8a9c2e1e4b4cfeac49d2cef6220`.
  Arşiv kopyasının yalnız JSON boşluk/satır biçimi normalize edilmiştir; parse edilmiş içerik
  özgün indirme ile alan-alan aynıdır.
- V2.4 config SHA-256: `ee3d2c1e76d096af406e3dc8b0e6a15b4e42297b8c05cd703692e4520c92cb71`
- V2.3 run-contract SHA-256: `6c4884d25da3f5d2af2b6d64522b3a78af08db5e05e9ce99e4b13d12686a5261`
- Doğrulanan artifact: 1.107 kayıt; 1.055 candidate artifact'ı hash ile doğrulandı.
- Görüntü/source group: 130.000 / 8.
- Yerel bağımsız yeniden hesaplama: 17/17 yapısal ve aritmetik kontrol geçti.
- Referans CSV, unblinded sonuç, exact manifest, controlled summary ve config hash'leri
  yerel kaynaklarla eşleşti.

## 3. Tam-kapsam sonucu

| Ölçüm | V2.3 | V2.4 |
|---|---:|---:|
| Aday çift | 4.621.932 | 2.491.553 |
| Çift/görüntü | 35,5533 | 19,1658 |
| Çıkarılan çift | - | 2.130.379 |

V2.4 yük kapısını geçti. Ancak bütün kapılar birlikte geçmedi:

| Kabul kapısı | Sonuç | Karar |
|---|---:|---|
| Exact SHA-256 | 1.393/1.393 | Geçti |
| Kontrollü pozitif | 1.040/1.040 | Geçti |
| Doğal aynı-kaynak | 22/23 | **Geçmedi** |
| Source group dışı çift | 0 | Geçti |
| Aday yükü | 19,1658 <= 25 | Geçti |

Bu nedenle `all_passed=false` ve v2.4 kabul edilmedi.

## 4. Source-group yükü

| Source group | V2.3 | V2.4 | V2.4 çift/görüntü | pHash |
|---|---:|---:|---:|---|
| all_blood_smear | 422.718 | 422.718 | 21,136 | açık |
| brain_ce_mri | 1.858.745 | 272.808 | 18,187 | kapalı |
| breast_histology | 277.878 | 277.878 | 27,788 | açık |
| cervix_pap_smear | 378.508 | 378.508 | 15,140 | açık |
| kidney_ct | 659.791 | 115.349 | 11,535 | kapalı |
| lung_colon_histology | 461.152 | 461.152 | 18,446 | açık |
| lymphoma_histology | 339.509 | 339.509 | 22,634 | açık |
| oral_histology | 223.631 | 223.631 | 22,363 | açık |

MRI ve CT'de pHash'in kapatılması beklenen 2.130.379 çifti kaldırdı. Exact güvenlik
ağı kidney CT'deki 75 ve lung/colon histolojideki 1.318 exact çifti korudu.

## 5. Kaçan doğal çift: N000123

Eksik çift:

- Pair ID: `N000123`
- Grup: `cervix_pap_smear`
- Leaf: `cervix_dyk` ↔ `cervix_dyk`
- Sol: `Cervical Cancer/cervix_dyk/cervix_dyk_4875.jpg`
- Sağ: `Cervical Cancer/cervix_dyk/cervix_dyk_3040.jpg`

6.604 gerçek görüntülük doğal-audit indeksindeki kayıt:

- pHash mesafesi: 34; dolayısıyla `pHash<=12` içinde değil.
- Robust top-15: evet; küçük audit indeksindeki en iyi birleşik sıra 1.
- Robust cosine: 0,919612765.
- ORB: 427 inlier, 0,910 inlier ratio.
- İkinci-aşama AI-destekli, uzman-olmayan adjudication: `same_source`.

Tam 130.000 görüntülük artifact üyelik testi çiftin hem v2.3 hem v2.4 birleşiminde
olmadığını gösterdi: ikisi de 22/23 doğal referans buldu. V2.4 serviks grubunda pHash'i
değiştirmediği için bu kayıp MRI/CT pHash kanalının kapatılmasından kaynaklanmıyor.

Kanıtların desteklediği sınırlı yorum şudur: küçük audit indeksinde robust rank 1 olan çift,
tam 25.000 görüntülük serviks havuzunda saklanan top-15 birleşimine giremedi. Bu, benzer ve
augment edilmiş komşular arttığında **rank crowding** oluşabileceğini gösterir. Gerçek full-index
robust sırası mevcut raporda yoktur; ölçülmeden kesin sıra iddia edilmemelidir.

## 6. Karar ve sonraki güvenli adım

- V2.4 config'i başarısız ablation olarak değişmeden korunur.
- V2.3 yük nedeniyle, v2.4 doğal-recall nedeniyle beklemede kalır.
- Final candidate listesi yazılmaz.
- 2,49 milyon çift scorer'a gönderilmez.
- Family, split ve classifier eğitimi başlamaz.

Önerilen bir sonraki adım, Mehmet'in onayından sonra mevcut descriptor artifact'larını
salt-okunur kullanan dar bir `N000123/full-cervix rank-crowding` tanısıdır. Bu tanı önce
N000123'ün gerçek full-index robust sırasını ve serviks için sınırlı top-K alternatiflerinin
aday yükünü ölçmelidir. Sonuç görülmeden v2.5 kuralı veya yeni K değeri seçilmemelidir.
