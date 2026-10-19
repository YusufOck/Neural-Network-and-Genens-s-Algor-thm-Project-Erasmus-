# Yakın-kopya / augmentation-family yöntem planı

Durum: **Planlandı; henüz uygulanmadı veya eşikleri doğrulanmadı.**

Tarih: 2026-10-19

Bu planın amacı, 130.000 görüntüde aynı kaynak görüntüden üretilmiş dosyaları bulmak ve bir
ailenin üyelerinin train/validation/test arasında bölünmesini engellemektir. Bu bir kanser
sınıflandırma modeli değildir ve henüz split oluşturmaz.

## 1. Mevcut 120 çiftin gerçek rolü

`brain_glioma` ve `lung_scc` pilotundaki 120 çift nihai kalibrasyon veya bütün veri setini temsil
eden test kümesi değildir. Çiftler pHash uzaklığı 0–12 aralığından ve yalnızca iki alt sınıftan
seçildiği için pHash'e ve bu iki görüntü türüne göre yanlıdır.

Bu 120 çift:

- pHash'in MRI ile histolojide farklı davrandığını gösteren bir **hata-analizi pilotudur**;
- Pair ID, temas sayfası ve insan inceleme şemasını denememizi sağlar;
- daha sonra kurulacak resmi kalibrasyon kümesine örnek sağlayabilir;
- tek başına eşik seçmek, recall ölçmek veya 26 alt sınıf için kural belirlemek için yeterli değildir.

Pilot kayıtları yarım bırakılmayacak; ancak önce resmi kalibrasyon tasarımı sabitlenecek, sonra
pilot kararları bu yeni şemaya göre tamamlanacaktır.

## 2. Veri setinde neden beş görüntü modalitesi var?

`Modalite`, görüntünün hangi fiziksel/teknik yöntemle üretildiğini anlatır; sınıf sayısı değildir.
Sekiz üst grup ve 26 alt sınıf, aşağıdaki beş temel görüntü türüne dağılır:

| Modalite | Ne gösterir? | Alt sınıflar | Sayı |
|---|---|---|---:|
| Periferik kan yayması mikroskopisi | Boyanmış tek kan/lenfoid hücreleri | `all_benign`, `all_early`, `all_pre`, `all_pro` | 4 |
| T1 kontrastlı MRI | Beynin kesit görüntüleri | `brain_glioma`, `brain_menin`, `brain_tumor` | 3 |
| H&E histopatoloji | Boyanmış doku kesitleri | breast 2, lung/colon 5, lymphoma 3, oral 2 | 12 |
| Pap smear hücre mikroskopisi | Rahim ağzından izole hücreler | Beş `cervix_*` sınıfı | 5 |
| Abdominal CT | Böbreğin X-ray tabanlı kesitleri | `kidney_normal`, `kidney_tumor` | 2 |

Toplam `4 + 3 + 12 + 5 + 2 = 26` alt sınıftır. H&E tek bir geniş modalite olsa da breast,
lung/colon, lymphoma ve oral görüntüleri farklı kaynak ve büyütmelerden gelir; doğrulamada bu
kaynak grupları ayrıca temsil edilecektir.

## 3. Seçilen yöntem: çok-aşamalı hibrit denetim

Tek bir pHash, ORB veya SSIM skoru nihai karar vermeyecektir. Aynı protokol bütün alt sınıflarda
uygulanacak; eşikler yalnız kalibrasyon kanıtı farklılık gösterirse modalite/kaynak grubuna göre
değişebilecektir.

### Aşama 0 — Birebir kopya kanıtı: SHA-256

- Byte-byte aynı dosyaları kesin olarak belirler.
- Mevcut tam veri denetimi korunur.
- SHA-256 eşitliği doğrudan `exact_duplicate` kararıdır.
- SHA-256 farklılığı, görüntülerin bağımsız olduğunu kanıtlamaz.

### Aşama 1 — Yüksek-recall aday üretimi

Amaç mümkün olduğunca az gerçek akraba kaçırmak; bu aşamada false positive kabul edilebilir.

1. Görüntüyü dosya uzantısına güvenmeden decode et.
2. Yapısal karşılaştırma için gri-ton sürüm üret; renkli mikroskopi için RGB bilgiyi ayrıca koru.
3. MRI/CT için büyük tekdüze siyah çerçeveyi maskele/kırp; aksi halde pHash anatomiden çok arka
   planı karşılaştırabilir.
4. pHash'i yalnız aday üretici olarak kullan.
5. Bilinen augmentation politikasına karşı orijinal ve yatay-çevrilmiş sürüm gibi kontrollü
   dönüşüm temsillerini aday aramada değerlendir; kabul kararı verme.
6. Bütün 130.000 görüntünün kör `N²` karşılaştırması yerine, 64-bit hash değerleri için Hamming
   yarıçapı indekslemesi/BK-tree veya eşdeğer bir aday indeksi kullan.
7. Aynı alt sınıf içi adaylara ek olarak, yalnız uyumlu modalite/kaynak gruplarında sınıflar arası
   aday üret.

pHash tek başına aile oluşturamaz. `brain_glioma` pilotu, uzaklık 2'de bile çok sayıda yanlış
aday bulunduğunu göstermiştir.

### Aşama 2 — Geometrik doğrulama

#### Dokulu görüntüler: ORB + descriptor matching + RANSAC affine

Histoloji, kan hücresi ve Pap hücrelerinde yerel noktalar yeterliyse:

- ORB ile ayırt edici yerel noktalar ve binary descriptor'lar çıkar;
- Hamming tabanlı eşleştirme ve ratio/cross-check filtrelemesi uygula;
- RANSAC ile affine dönüşüm tahmin et;
- iyi eşleşme sayısı, geometrik inlier sayısı, inlier oranı ve görüntüye yayılımını kaydet.

ORB, yönlendirilmiş ve çok-ölçekli yerel özellikler üretir; RANSAC ise hatalı eşleşmeler arasından
tek bir tutarlı geometrik dönüşümü destekleyen inlier'ları ayırır. Kaynaklar:

- [OpenCV ORB belgesi](https://docs.opencv.org/4.13.0/d1/d89/tutorial_py_orb.html)
- [OpenCV feature matching ve RANSAC homography belgesi](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)

Belgelenen augmentation'lar rotation, shift, shear ve zoom olduğu için birincil dönüşüm modeli
affine olacaktır. Homography yalnız kontrollü bir fallback/diagnostic olarak denenecek; aşırı
esnek bir dönüşümün bağımsız görüntüleri yanlış eşleştirmesine izin verilmeyecektir.

#### Az yerel özellikli MRI/CT: ECC/phase-correlation fallback

MRI ve CT'de siyah arka plan veya yumuşak doku nedeniyle ORB yeterli nokta bulamayabilir. Bu
durumda:

- kaba translation için phase correlation;
- intensity tabanlı affine hizalama için `findTransformECC`;
- geçerli foreground maskeleri ve dönüşüm-plausibility sınırları

denenecektir. ECC bir “aynı kaynak” kararı değil, iki adayın hizalanabilirliğini ölçen fallback
olacaktır.

- [OpenCV `findTransformECC` belgesi](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
- [OpenCV `phaseCorrelate` belgesi](https://docs.opencv.org/4.13.0/d7/df3/group__imgproc__motion.html)

### Aşama 3 — Hizalama sonrası yapısal doğrulama

Geometrik dönüşümle hizalanan ortak/geçerli bölgede:

- SSIM: yerel yapı, kontrast ve doku benzerliği;
- NMI: özellikle MRI/CT gibi yoğunlukların değişebildiği görüntülerde ortak bilgi;
- normalize hata ve geçerli-overlap oranı: destekleyici kalite kontrolleri

hesaplanacaktır. Ham, hizalanmamış SSIM nihai karar olarak kullanılmayacaktır.

- [scikit-image SSIM ve NMI belgeleri](https://scikit-image.org/docs/stable/api/skimage.metrics.html)

### Aşama 4 — Birleşik karar ve insan incelemesi

Bir çiftin otomatik `same_source_candidate` kabul edilmesi için tek bir düşük pHash değeri yeterli
olmayacaktır. Planlanan karar girdileri:

- candidate kaynağı ve hash uzaklıkları;
- ORB iyi eşleşme/inlier/coverage ölçümleri;
- ECC/phase-correlation sonucu ve tahmin edilen dönüşümün mantıklılığı;
- hizalanmış SSIM/NMI/overlap;
- modalite ve kaynak grubu;
- SHA-256 exact bilgisi.

Karar sınıfları:

- `exact_duplicate`: SHA-256 ile kesin;
- `same_source_candidate`: birden fazla bağımsız kanıtla güçlü aday;
- `independent_lookalike`: benzer sınıf/modality fakat aynı kaynak kanıtı yok;
- `uncertain`: insan incelemesi veya daha güçlü yöntem gerekli.

Eşikler elle keyfi seçilmeyecek. Kalibrasyon çiftlerinin bir bölümünde belirlenecek, farklı kaynak
görüntülerden oluşan hold-out çiftlerinde ölçülecektir.

### Opsiyonel fallback — öğrenilmiş copy-detection descriptor

Yukarıdaki hafif yöntemler kontrollü pozitiflerde yeterli recall sağlamazsa, genel sınıflandırma
embeddingi yerine image-copy detection için geliştirilmiş bir descriptor ayrı bir ablation olarak
değerlendirilebilir. Örneğin SSCD çalışması, kopyaları yakınlaştırırken yalnız semantik olarak
benzer görüntüleri ayırmayı amaçlar; ancak kısmi kopyalarda yerel yöntemlerin gerekebileceğini de
belirtir:

- [SSCD — A Self-Supervised Descriptor for Image Copy Detection, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.html)

Bu yöntem başlangıç bağımlılığı değildir; yalnız temel hibrit yöntem yetersiz kalırsa denenir.

## 4. Resmi kalibrasyon kümesi nasıl kurulacak?

Mevcut 120 pHash çifti yerine/yanına bütün 26 alt sınıfı kapsayan kontrollü bir çift veri seti
kurulacaktır.

### Bilinen pozitifler

1. SHA-256 exact çiftler.
2. Her 26 alt sınıftan seed ile seçilmiş görüntülere, kaynak README'de belgelenen işlemler
   uygulanarak oluşturulan sentetik akrabalar:
   - ±10 derece rotation;
   - %10 width/height shift;
   - shear ve zoom;
   - horizontal flip;
   - brightness değişimi.
3. pHash ve diğer aday üreticilerinin bulduğu, insan tarafından doğrulanmış doğal akrabalar.

Sentetik görüntüler yalnız denetim yöntemini kalibre etmek için kullanılacak; sınıflandırma modelinin
train/validation/test verisine eklenmeyecektir.

### Bilinen/kuvvetli negatifler

1. Her alt sınıftan rastgele farklı görüntüler.
2. Aynı sınıfta görsel olarak çok benzeyen “hard negative” çiftler.
3. Aynı modalitedeki farklı alt sınıflardan çapraz-etiket çiftler.

Hasta/slide ID olmadığı için rastgele negatifler otomatik olarak kesin bağımsız kabul edilmeyecek;
şüpheli olanlar `uncertain` bırakılacaktır.

### Recall sınaması

- Yalnız pHash 0–12 çiftleri değil, 12 üzerindeki bantlar ve diğer aday kanalları da örneklenecek.
- Kalibrasyon ve hold-out doğrulama çiftleri aynı kaynak görüntünün sentetik akrabalarını paylaşmayacak.
- Aday üretim recall'u, doğrulama precision'ı, false-negative oranı ve insan incelemesine kalan oran
  modalite ve kaynak grubu başına raporlanacak.

Planlanan kabul hedefi, controlled/sentetik pozitiflerde her modalite için en az %99 candidate
recall'dur; bu yalnız hedeftir, henüz elde edilmiş sonuç değildir. Otomatik aile kararı için yüksek
precision ve incelenmiş false-positive örnekleri ayrıca gerekli olacaktır.

## 5. Modaliteye göre planlanan yöntem ağırlığı

| Modalite | Aday üretimi | Birincil doğrulama | Destek/fallback |
|---|---|---|---|
| Kan yayması | pHash + dönüşüm varyantları | ORB + affine RANSAC | hizalanmış SSIM, renk desteği |
| MRI | foreground-maskeli pHash | ECC/phase correlation + NMI/SSIM | ORB inlier kanıtı varsa destek |
| H&E histopatoloji | pHash + flip-aware adaylar | ORB + affine RANSAC | hizalanmış SSIM, renk/overlap desteği |
| Pap smear | foreground/object crop + pHash | ORB + affine RANSAC | ECC ve hizalanmış SSIM |
| CT | foreground-maskeli pHash | ECC/phase correlation + NMI/SSIM | ORB inlier kanıtı varsa destek |

Bu tablo nihai eşikleri değil, hangi kanıt türüne öncelik verileceğini belirler. Kalibrasyon sonucu
bir yöntem bir modalitede başarısız olursa plan güncellenecektir.

## 6. Bütün veri setine uygulama ve split ön koşulu

Kalibrasyon kabul koşullarını geçtikten sonra:

1. Bütün 130.000 görüntünün gerekli descriptor/hash değerleri bir kez hesaplanır ve kaydedilir.
2. Aynı alt sınıf içi ve uyumlu modalitelerde sınıflar arası adaylar indeksle bulunur.
3. Pahalı ORB/ECC/SSIM yalnız aday çiftlerde çalıştırılır.
4. Kabul edilen ilişkiler bir graph oluşturur; connected component içindeki tüm görüntüler aynı
   augmentation family sayılır.
5. Her ailenin dosyaları tek bir train/validation/test bölümünde tutulur.
6. Split öncesi ve sonrası aile kesişim testi yapılır.

Bu altı adım bitmeden sabit 70/15/15 split oluşturulmayacaktır.

## 7. Bir sonraki onay noktası

Henüz kod veya yeni tarama çalıştırılmayacak. Mehmet bu yöntem zincirini anladıktan/onayladıktan
sonra ilk uygulama adımı, bütün 26 alt sınıftan kontrollü pozitif ve negatif çift üreten
`calibration-set v1` taslağı olacaktır. Mevcut 120 çift de ayrı `legacy pHash pilot` etiketiyle
korunacaktır.
