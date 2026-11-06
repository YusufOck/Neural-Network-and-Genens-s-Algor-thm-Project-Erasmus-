# Candidate Generator v2.4 modality-aware karşı-olgusal planı

Tarih: 2026-11-06

Durum: **Kural tam artifact değerlendirmesinden önce JSON + SHA-256 ile donduruldu.
Analiz kodu ve yerel testleri hazır; 130.000 görüntülük mevcut v2.3 artifact'ları üzerinde
salt-okunur değerlendirme henüz çalıştırılmadı.**

## 1. Bu adımın amacı

V2.3, 130.000 görüntüde 4.621.932 aday (35,553 çift/görüntü) üretti ve önceden
belirlenen 25 sınırında güvenli biçimde durdu. Yükün ana nedeni global `pHash<=12`
kuralının brain MRI ve kidney CT görüntülerinde çok fazla benzer görünümlü aday
üretmesidir.

Bu çalışma yeni görüntü descriptor'ı veya pHash üretmez. Drive'daki tamamlanmış ve
hash-kayıtlı 1.107 v2.3 artifact'ını salt-okunur biçimde kullanarak şu soruyu test eder:

> MRI ve CT'de pHash'i kapatıp robust top-15 ile exact SHA-256 güvenlik ağını korursak,
> referans pozitifleri kaçırmadan aday yükünü 25 çift/görüntünün altına indirebilir miyiz?

Bu bir duplicate kararı, family oluşturma, veri split'i veya model eğitimi değildir.

## 2. Önceden dondurulan v2.4 hipotezi

Kaynak: `config/candidate_generator_v2_4_counterfactual.json`

Config SHA-256:

`ee3d2c1e76d096af406e3dc8b0e6a15b4e42297b8c05cd703692e4520c92cb71`

Kural:

1. Robust dört-kanal ResNet18 top-15 adayları sekiz source group'un tamamında korunur.
2. Flip-aware `pHash<=12` şu altı grupta korunur: ALL kan yayması, breast histology,
   cervix Pap smear, lung/colon histology, lymphoma histology ve oral histology.
3. `pHash<=12`, `brain_ce_mri` ve `kidney_ct` gruplarında kapatılır.
4. Bilinen SHA-256 exact çiftleri bütün gruplarda zorunlu olarak birleşime eklenir.
5. Source group dışı candidate yasaktır.

## 3. Önceden belirlenen kabul kapıları

| Kapı | Gerekli sonuç |
|---|---:|
| Exact SHA-256 çiftleri | 1.393/1.393 |
| Doğal aynı-kaynak referansları | 23/23 |
| Kontrollü sentetik pozitifler | 1.040/1.040 |
| Source group dışı çift | 0 |
| Candidate yükü | en fazla 25 çift/görüntü |

Bir kapı bile geçmezse v2.4 kabul edilmeyecek ve v2.3 beklemede kalacaktır.

## 4. Tam çalışmadan önce doğrulanan dayanak

Ham kontrollü metrik artifact'ı Git'e eklenmez. Onun yerine dosyanın SHA-256'sını ve
grup bazlı sayımları taşıyan
`research/candidate_generator_v2_4_controlled_reference_summary.json` kaydedildi.
Yerel test bu özeti ham 1.040 satırlık CSV'den yeniden hesaplayarak doğruladı.

| Referans grubu | Referans | Robust top-15 buldu | V2.4 buldu |
|---|---:|---:|---:|
| ALL kan yayması | 160 | 156 | 160 |
| Brain CE-MRI | 120 | 120 | 120 |
| Breast histology | 80 | 80 | 80 |
| Cervix Pap smear | 200 | 200 | 200 |
| Kidney CT | 80 | 80 | 80 |
| Lung/colon histology | 200 | 200 | 200 |
| Lymphoma histology | 120 | 120 | 120 |
| Oral histology | 80 | 80 | 80 |
| **Toplam** | **1.040** | **1.036** | **1.040** |

MRI ve CT kontrollü pozitiflerinin 200/200'ü robust top-15 tarafından bulunduğu için
bu iki grupta pHash'in kaldırılması bu kontrollü referansları kaybetmez. Robust kanalın
kaçırdığı dört kontrollü örnek ALL kan yayması grubundadır; v2.4 bu grupta pHash'i
koruduğu için 1.040/1.040 toplam korunur.

Nihai doğal audit referansındaki 23 aynı-kaynak çiftin 23/23'ü de robust top-15
alanındadır. Buna `brain_ce_mri` içindeki `N000239` da dahildir. Tam değerlendirme
bu 23 çiftin gerçek full-index pair key'lerini yeni birleşimde ayrıca kontrol edecektir.

## 5. Analiz aracının güvenlik özellikleri

`src/analyze_candidate_generator_v2_4_counterfactual.py`:

- v2.4 config hash'ini doğrular;
- Git/Colab ve Windows arasında bilimsel içeriği değiştirmeyen CRLF/LF farklarını
  UTF-8 metni LF'ye normalize ederek doğrular;
- v2.3 run-contract ve 1.107 stage envanterini doğrular;
- kullandığı image-index, 32 robust ve 1.022 pHash retrieval artifact'ının SHA-256
  değerlerini kontrol eder;
- mevcut artifact'ları değiştirmez;
- raporun v2.3 klasörünün içine yazılmasını reddeder;
- final candidate CSV/NPZ üretmez;
- duplicate scorer, family, split veya classifier çalıştırmaz;
- doğal, kontrollü ve exact kanıt sonuçlarını ayrı bölümlerde raporlar.

## 6. Beklenen fakat henüz sonuç olmayan kapasite

V2.3 tanısından yalnız MRI/CT pHash-only yükü çıkarıldığında matematiksel beklenti
yaklaşık 2.491.553 aday veya 19,166 çift/görüntüdür. Exact çiftlerin zorunlu
birleşimi nedeniyle gerçek sayı mevcut artifact'lar okunmadan kesin sonuç olarak
yazılamaz.

Bu yüzden 19,166 yalnız bir beklentidir. Rapor üretildikten ve hash/kapılar
doğrulandıktan sonra `reproduced` sonuç sayılabilir.

## 7. Kesinti ve kredi güvenliği

- Kod, config, test ve bu plan tam Colab çalışmasından önce GitHub'a push edilir.
- Tam değerlendirme GPU veya görüntü yeniden işleme istemez; mevcut küçük NPZ candidate
  artifact'larını okuyan CPU ağırlıklı bir işlemdir.
- V2.3 klasörü salt-okunur tutulur; kesinti eski 1.107 artifact'ı bozmaz.
- Rapor yalnız bütün kontroller tamamlanırsa atomik olarak yeni bir dosyaya yazılır.
- Colab kesilirse aynı komut tekrar çalıştırılabilir; pahalı descriptor/retrieval
  aşamaları tekrarlanmaz.

## 8. Bu planın yetki vermediği işler

V2.4 bütün kapıları geçse bile ayrıca inceleme ve Mehmet'in yeni onayı olmadan:

- final candidate listesi yazılmayacak;
- 2,5 milyon civarı çift scorer'a gönderilmeyecek;
- duplicate family oluşturulmayacak;
- train/validation/test split hazırlanmayacak;
- model eğitilmeyecek.
