# Candidate Generator v2.3 Colab/T4 Smoke Doğrulaması

Tarih: 2026-11-06

## Amaç ve kapsam

Bu çalışma yalnız Candidate Generator v2.3'ün Google Colab/T4 ortamında doğru
kurulduğunu ve hash-korumalı resumable artifact zincirini üretebildiğini sınayan bir
**altyapı smoke testidir**. Duplicate kararı, family oluşturma, train/validation/test
split veya sınıflandırıcı eğitimi yapılmadı. Sonuçlar model accuracy/F1 metriği değildir.

Dataset `MyDrive/NNGA_DATA/Multi Cancer.zip` arşivinden Colab'ın geçici
`/content/NNGA_DATA` diskine 4,7 dakikada çıkarıldı. Arşiv 8,62 GiB, çalıştırma
başlangıcındaki yerel boş alan 112,54 GiB idi. Exact-duplicate manifesti yeniden
üretildi ve yerel audit ile aynı değerleri verdi:

- kabul edilen görüntü: 130.000;
- exact-duplicate grubu: 1.318;
- gruplardaki dosya: 2.673;
- redundant exact kopya: 1.355;
- alt sınıflar arası exact grup: 0.

## Çalıştırma sözleşmesi

- Git commit: `3528c83b0923f0e237d11c54875b2b7ab9d295fd`
- Scope: `smoke`
- Seed: `42`
- Seçim: 26 leaf × 2 görüntü = 52 görüntü
- Frozen config SHA-256:
  `daef2f4d1c4199846a64b549a5070f5c8cc3f322cc949fb55dde2700ef2e3776`
- Run contract SHA-256:
  `63c32741a7b6961027c39341fd3c6d09eea8fd804865a35074fa85f82cf090dc`
- Normalize exact-manifest SHA-256:
  `e8a27465513e948d2cf7a012a4d2eaebea2dd97d8f5c313299f81eb86780ee4a`

Ortam:

- Python 3.13.16;
- PyTorch 2.11.0+cu130;
- torchvision 0.26.0+cu130;
- NumPy 2.4.6;
- Pillow 11.3.0;
- CUDA 13.0;
- Tesla T4.

## Sonuç

Durum: `candidate_generator_v2_3_resumable_smoke_complete`

| Ölçüm | Değer |
|---|---:|
| Görüntü | 52 |
| Robust aday | 166 |
| pHash adayı | 1 |
| Kesişim | 1 |
| Birleşik aday | 166 |
| Aday/görüntü | 3,1923 |
| Cross-source-group çift | 0 |
| Smoke kapsamındaki beklenen exact çift | 0 |
| Eksik exact çift | 0 |
| Rapor dışı kayıtlı artifact | 95 |

Source-group aday sayıları:

- ALL blood smear: 28;
- brain CE-MRI: 15;
- breast histology: 6;
- cervix Pap smear: 45;
- kidney CT: 6;
- lung/colon histology: 45;
- lymphoma histology: 15;
- oral histology: 6.

Birikimli stage süreleri:

- index: 0,011 saniye;
- pHash üretimi: 0,750 saniye;
- descriptor üretimi: 16,670 saniye;
- robust retrieval: 0,548 saniye;
- pHash retrieval: 0,137 saniye;
- finalization: 0,021 saniye.

Bu süreler 52 görüntülük küçük smoke'un başlangıç overhead'ini içerir ve 130.000
görüntülük süreye doğrusal biçimde genellenmemelidir.

## Bağımsız artifact doğrulaması

Kullanıcının Drive'dan indirdiği `run_report.json` ve `hash_manifest.json` üzerinde
15/15 kontrol geçti:

- indirilen rapor SHA-256'sı manifest kaydıyla eşleşti:
  `995747e39643b1c81fa399d4c3d55b5e59c03a2fb1b15b7ccbb8e414761d626e`;
- run-contract hash'i iki dosyada aynı;
- frozen config hash'i yerel config dosyasıyla aynı;
- 26 leaf'in her birinde tam iki görüntü ve toplam 52 görüntü var;
- sekiz source-group aday toplamı 166;
- robust/pHash/kesişim küme aritmetiği birleşik 166 sonucu veriyor;
- aday/görüntü oranı yeniden hesaplandığında aynı;
- canonical çift sayısı 166, cross-group çift 0;
- normalize exact-manifest hash'i beklenen değerle aynı;
- manifest 95 rapor-dışı + bir rapor olmak üzere 96 hash girdisi içeriyor;
- 26 descriptor, 26 pHash, 32 robust-retrieval ve sekiz pHash-blok artifact'ı
  eksiksiz kayıtlı;
- bütün artifact hash değerleri geçerli 64 karakterli SHA-256 biçiminde;
- CUDA ve Tesla T4 kaydı mevcut.

## Yorum ve sınırlar

Smoke altyapı kapısını geçti. Ancak 52 görüntülük seçimde beklenen exact çift sayısı
sıfırdır; bu nedenle `missing_exact_duplicate_pairs=0`, exact-duplicate recall kanıtı
değildir. Frozen candidate kuralının retrieval kanıtı önceki kontrollü ve doğal audit
çalışmalarından gelir. Ayrıca 166 adayın hiçbiri otomatik olarak duplicate kabul
edilmez. Full retrieval tamamlanırsa adaylar frozen duplicate rule v1 ile ayrı,
resumable CPU shard'larında skorlanmalıdır.

## Karar noktası

Colab/T4 ortamı tam 130.000 görüntülük candidate retrieval için teknik olarak uygun
bulundu. Full retrieval henüz başlamadı ve ayrı kullanıcı onayı gerektiriyor. Bu işlem
tamamlansa bile family, split veya model eğitimi otomatik başlamayacaktır.
