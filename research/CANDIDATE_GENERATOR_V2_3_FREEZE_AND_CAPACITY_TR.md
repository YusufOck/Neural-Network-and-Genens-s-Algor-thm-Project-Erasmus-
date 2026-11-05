# Candidate Generator v2.3 freeze ve kapasite planı

Tarih: 2026-11-04

Durum: **V2.3 config, yöntem bağımlılıkları ve ResNet18 checkpoint hash'iyle
donduruldu ve yerelde doğrulandı. 130.000 görüntülük retrieval henüz
başlatılmadı.** Duplicate family, train/validation/test split veya model eğitimi
yoktur.

## 1. “Dondurmak” ne demektir?

Dondurmak, tam veri sonucunu gördükten sonra kuralı sessizce değiştirmemek için
yöntemi çalıştırmadan önce kesinleştirmektir. V2.3 için aşağıdakiler sabittir:

- robust kanal: dört ResNet18 descriptor kanalının yönlü top-15 komşularının
  birleşimi;
- pHash güvenlik kanalı: orijinal ve yatay çevrilmiş hash'lerle Hamming
  uzaklığı `<=12`;
- nihai aday listesi: robust kanal ile pHash kanalının tekrarsız birleşimi;
- arama alanı: yalnız uyumlu source group içi;
- aynı source group içindeki farklı leaf sınıfları karşılaştırılabilir;
- farklı modalite/source group çiftleri üretilmez.

Bu aşama yalnız “hangi çiftleri ayrıntılı duplicate doğrulamasına göndereceğiz?”
sorusunu cevaplar. Bir aday çift otomatik olarak duplicate değildir.

## 2. Dondurulmuş kimlik

| Öğe | SHA-256 |
|---|---|
| `config/candidate_generator_v2_3.json` | `daef2f4d1c4199846a64b549a5070f5c8cc3f322cc949fb55dde2700ef2e3776` |
| ResNet18 ImageNet checkpoint | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Natural final references | `e9a2336ab954926adeafb891f7b3efc8963c149385d09c523fd4c130b9c21acd` |

Config ayrıca descriptor, retrieval, pHash, veri indeksleme, source-group ve
frozen duplicate-rule kaynak dosyalarının yedi SHA-256 değerini taşır. Bunlardan
biri değişirse tam çalışma başlamadan doğrulama durur.

## 3. Yerel freeze doğrulaması

`src/verify_candidate_generator_v2_3_freeze.py` aşağıdakileri doğruladı:

- 130.000 kabul edilen görüntü;
- 26 leaf × 5.000 görüntü;
- 8 uyumlu source group;
- v2.3 top-15 ve pHash `<=12` ayarları;
- yedi kod/config bağımlılığının hash'i;
- ResNet18 checkpoint hash'i;
- 345 doğal referans içinde 23 aynı-kaynak çiftin tamamının candidate alanında
  olması ve outside aynı-kaynak sayısının 0 kalması;
- v2.2 başarısız sonucunun değiştirilmeden kanıt zincirinde tutulması.
- Python 3.10 proje testleri: **85/85 geçti**.

Makine-okunur çıktı:
`research/candidate_generator_v2_3_freeze_report.json`.

## 4. Tam çalışma kapasite tahmini

Bu değerler tam çalışma ölçümü değil, daha küçük doğrulanmış çalışmalardan lineer
kapasite tahminidir.

| Kaynak | Tahmin |
|---|---:|
| V2.3 aday yoğunluğu | 22,501 çift/görüntü |
| 130.000 görüntü için aday çift | yaklaşık 2.925.170 |
| Doğrulama öncesi durdurma bütçesi | 3.250.000 çift |
| Dört float32 descriptor matrisi | 1.064.960.000 bayt ≈ 0,99 GiB |
| İki uint64 pHash dizisi | 2.080.000 bayt ≈ 1,98 MiB |
| Frozen duplicate scorer seri süre | yaklaşık 27,59 gün |
| Sekiz worker için teorik alt sınır | yaklaşık 3,45 gün |

Sekiz-worker değeri ideal lineer hızlanma varsayar. Gerçekte disk erişimi,
OpenCV/ECC maliyeti, süreçler arası veri aktarımı ve zor görüntüler nedeniyle
daha uzun sürebilir. Bu yüzden yaklaşık üç milyon çift doğrudan tek süreçte
başlatılmayacaktır.

## 5. Yerel bilgisayar ve Colab görev ayrımı

### Colab/T4 için uygun bölüm

- 130.000 görüntünün dört-kanallı ResNet18 descriptor çıkarımı;
- source-group içi exact cosine top-15 retrieval;
- pHash kanalının hesaplanması ve iki aday listesinin birleşimi;
- ara çıktıları Google Drive'a hash'leriyle kaydetmek.

V2.2'de 7.644 görüntünün descriptor işlemi RTX 3050'de yaklaşık 550 saniye
sürdü. Tam veri için lineer kaba ölçek yaklaşık 2,6 saattir; Drive I/O ve Colab
ortamı nedeniyle pratik süre daha uzun olabilir. Bu değer tam çalışma garantisi
değildir.

### CPU/shard planı gereken bölüm

- aday çiftlerde ORB/RANSAC, phase correlation, uygun modalitelerde ECC;
- hizalanmış SSIM/NMI/korelasyon;
- frozen duplicate rule v1 kararı;
- `same_source`, `independent`, `uncertain` sonuçlarının birleştirilmesi.

Bu bölüm GPU'dan çok CPU ve disk erişimi kullanır. Önce küçük, sabit bir aday
shard'ında gerçek worker hızı ölçülecek; ardından shard sayısı ve çalıştırma
ortamı belirlenecektir.

## 6. Tam çalışmadan önce güvenlik kapıları

Tam retrieval çıktısı şu koşulları sağlamazsa duplicate doğrulamasına geçilmez:

1. Envanter tam olarak 130.000/26/8 olmalı.
2. Config, kod bağımlılıkları ve checkpoint hash'leri eşleşmeli.
3. Candidate çiftleri benzersiz ve `left_index < right_index` biçiminde olmalı.
4. Source group dışı çift sayısı 0 olmalı.
5. Bilinen SHA-256 exact-duplicate çiftlerinin tamamı aday birleşiminde bulunmalı.
6. Aday yükü 3.250.000'i aşarsa doğrulama başlamadan durup kapasite yeniden
   değerlendirilmeli.
7. Var olan final çıktıların üzerine yazılmamalı; yalnız hash-doğrulanmış ara
   aşamalardan devam edilmeli.

## 7. Sıradaki kontrollü adım

Bir sonraki adım, frozen config'i değiştirmeden kullanan Colab/local orchestration
notebook'u hazırlamaktır. Notebook önce küçük bir throughput shard'ı çalıştıracak;
çıktı şeması ve hash'ler doğrulandıktan sonra kullanıcı açıkça onay verirse tam
130.000 görüntülük retrieval başlatılacaktır.

Tam retrieval sonunda Codex'e geri verilmesi planlanan dosyalar:

- deterministik `image_index.csv`;
- `phash_variants.npz`;
- dört kanallı `robust_descriptors.npz`;
- benzersiz `candidate_pairs.csv`;
- süre, cihaz, sayım ve hash'leri içeren `run_report.json`;
- teslim edilen dosyaların `hash_manifest.json` kaydı.

Tam retrieval tamamlanmış olsa bile henüz split veya model eğitimi başlamaz;
önce aday çiftlerin frozen duplicate rule ile doğrulanması ve family bileşenlerinin
oluşturulması gerekir.
