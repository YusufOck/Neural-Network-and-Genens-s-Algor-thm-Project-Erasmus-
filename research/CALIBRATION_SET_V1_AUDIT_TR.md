# Duplicate calibration-set v1 üretim ve doğrulama raporu

Durum: **Bütün 26 alt sınıfı kapsayan taslak üretildi; insan incelemesi ve hibrit yöntem
doğrulaması henüz tamamlanmadı.**

Tarih: 2026-10-19

## 1. Bu çıktı nedir, ne değildir?

Bu çıktı, aynı kaynak görüntüden üretilmiş dönüşümleri bulacak yakın-kopya yöntemini geliştirmek ve
ayrı bir hold-out bölümünde sınamak için hazırlanmış bir **audit kalibrasyon kümesidir**. Kanser
sınıflandırma modelinin train/validation/test split'i değildir; model eğitimi yapılmamıştır.

Üretim komutu:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\build_duplicate_calibration_set.py --seed 42 --workers 4
```

Kod, mevcut çıktı klasörünü güvenlik amacıyla ezmez. Yerel ve Git tarafından yok sayılan çıktılar:

- `artifacts/audit/duplicate_calibration_v1/pairs.csv`
- `artifacts/audit/duplicate_calibration_v1/review_candidates.csv`
- `artifacts/audit/duplicate_calibration_v1/report.json`
- `artifacts/audit/duplicate_calibration_v1/generated/`

## 2. Doğrulanmış üretim sayıları

| İçerik | Sayı | Anlamı |
|---|---:|---|
| Toplam çift | 1.342 | Bütün manifest |
| Kontrollü sentetik pozitif | 1.040 | Sol görüntüden kodla üretildiği için aynı kaynak olduğu kesin |
| SHA-256 exact pozitif | 24 | Byte-byte aynı olduğu için aynı kaynak olduğu kesin |
| Rastgele aynı-yaprak inceleme adayı | 104 | Negatif olduğu varsayılmadı; insan kararı bekliyor |
| pHash-hard aynı-yaprak inceleme adayı | 104 | Düşük pHash mesafeli; insan kararı bekliyor |
| Aynı kaynak-grubu çapraz-yaprak adayı | 70 | Farklı etiketli olası ilişki/hata kontrolü; insan kararı bekliyor |
| Toplam insan inceleme adayı | 278 | `human_verdict` alanlarının tamamı bilinçli olarak boş |
| Üretilen PNG | 1.040 | Yaklaşık 269,99 MiB; yalnız audit artefaktı |

Exact pozitifler, tam veri denetiminde exact grubu bulunan altı yapraktan en çok dörder tane
alındığı için 24 adettir. Exact grubu bulunmayan yapraklar için kopya uydurulmamıştır; onların kesin
pozitifleri kontrollü sentetik dönüşümlerden gelir.

Her iki audit bölümünde (`development`, `holdout`) 26 yaprağın tamamı vardır. Her yaprak/bölümde:

- iki farklı kaynak görüntü;
- her kaynak için on kontrollü dönüşüm;
- toplam 20 sentetik pozitif

bulunur. Böylece `26 × 2 bölüm × 2 kaynak × 10 dönüşüm = 1.040` sentetik pozitif elde edilmiştir.

## 3. Kontrollü dönüşümler

Kaynak README'de bildirilen sınırları kapsamak için şu deterministik dönüşümler üretildi:

- −10° ve +10° rotation;
- +%10 yatay / −%10 dikey shift;
- 0,10 x-shear;
- 0,90 zoom-out ve 1,10 zoom-in;
- horizontal flip;
- 0,20 ve 1,20 brightness;
- rotation, shift, shear, zoom, flip ve brightness içeren birleşik stres örneği.

PIL uygulaması bu sınırlar içinde kontrollü bir surrogate'tur; Keras `ImageDataGenerator` çıktısını
pixel-pixel yeniden ürettiği iddia edilmez. Sentetik görüntüler sınıflandırma eğitimine eklenmeyecek,
yalnız duplicate yöntemini ölçmek için kullanılacaktır.

## 4. pHash 0–12 pilot sınırının somut recall problemi

Sentetik pozitifler aynı kaynaktan geldikleri için kaçırılmamaları gereken kontrollü örneklerdir.
Özel 64-bit pHash sonucunda:

- minimum mesafe: 0;
- maksimum mesafe: 44;
- mesafe 12 üzerindeki kesin pozitif: **552 / 1.040 (%53,08)**;
- yalnız `pHash <= 12` ile aday kalabilen kesin pozitif: **488 / 1.040 (%46,92)**.

Bu sonuç, eski pilotta kullanılan 0–12 aralığının aile kararı veya tek aday kapısı olamayacağını
doğrudan gösterir. Bu oran doğal veri setindeki gerçek kopya prevalansı değildir; yalnız kontrollü
dönüşümlerde pHash aday recall sınamasıdır.

| Dönüşüm | Çift | Ortalama pHash | Min–maks | `>12` |
|---|---:|---:|---:|---:|
| Brightness 0,20 | 104 | 0,75 | 0–2 | 0 |
| Brightness 1,20 | 104 | 1,90 | 0–10 | 0 |
| Combined stress | 104 | 30,54 | 20–40 | 104 |
| Horizontal flip | 104 | 31,44 | 26–36 | 104 |
| Rotation −10° | 104 | 16,62 | 6–24 | 84 |
| Rotation +10° | 104 | 16,19 | 4–26 | 77 |
| Shear 0,10 | 104 | 6,90 | 0–18 | 2 |
| Shift +%10/−%10 | 104 | 29,94 | 16–44 | 104 |
| Zoom 1,10 | 104 | 10,92 | 4–18 | 28 |
| Zoom 0,90 | 104 | 12,87 | 4–24 | 49 |

Modalite bazında 12 üzerindeki kontrollü pozitif oranları da farklıdır:

| Modalite | Kontrollü pozitif | `>12` | Oran |
|---|---:|---:|---:|
| Kan yayması | 160 | 85 | %53,1 |
| MRI | 120 | 47 | %39,2 |
| H&E histopatoloji | 480 | 276 | %57,5 |
| Pap smear | 200 | 111 | %55,5 |
| CT | 80 | 33 | %41,2 |

Bu farklılık tek global pHash eşiğinin neden savunulamayacağına ek kanıttır.

## 5. Partition güvenlik kontrolleri

Script ve bağımsız çıktı sayımı şu invariants sonuçlarını verdi:

- seçilmiş orijinal dosyanın development/holdout kesişimi: **0**;
- seçilmiş SHA-256 grubunun development/holdout kesişimi: **0**;
- her iki bölümde 26/26 yaprak: **evet**;
- tekrar eden Pair ID: **0**;
- önceden doldurulmuş insan kararı: **0 / 278**.

Bu yalnız seçilen orijinal yolların ve bilinen exact ailelerin ayrıldığını kanıtlar. Veri setinde
henüz keşfedilmemiş doğal augmentation aileleri olabileceği için gerçek source-family ayrılığı bu
aşamada garanti edilemez. Zaten kalibrasyonun amacı bu aileleri bulacak yöntemi doğrulamaktır.

## 6. Görsel kalite kontrolü

Kan yayması, MRI, H&E, Pap smear ve CT için orijinal/`combined_stress` örnekleri yan yana incelendi.
Her beşinde de aynı anatomik/hücresel içerik korunurken rotation, shift, flip ve brightness etkisi
görünür durumdaydı. Bazı örneklerde dolgu/interpolasyon sınırları belirgindir; bunlar kontrollü stres
testinin parçasıdır ve doğal duplicate bulunduğu şeklinde yorumlanmaz.

## 7. Henüz yapılmayanlar ve sonraki kontrollü adım

- 278 review adayı henüz insan tarafından etiketlenmedi.
- Eski iki-sınıflı pilotun 112 bekleyen kararı henüz kapatılmadı.
- ORB/RANSAC, ECC/phase-correlation veya hizalanmış SSIM/NMI bu kümede çalıştırılmadı.
- Hiçbir modalite eşiği seçilmedi.
- 130.000 görüntüde doğal yakın-kopya taraması veya aile oluşturma yapılmadı.
- 70/15/15 sınıflandırma split'i oluşturulmadı.

Bir sonraki adım, 278 yeni review çiftine kalıcı Pair ID'leriyle temas sayfaları üretmek ve insan
karar şemasını birlikte kontrol etmektir. Ondan sonra hibrit doğrulayıcı kodlanıp development
bölümünde ayarlanacak ve holdout'a yalnız bir kez uygulanacaktır.
