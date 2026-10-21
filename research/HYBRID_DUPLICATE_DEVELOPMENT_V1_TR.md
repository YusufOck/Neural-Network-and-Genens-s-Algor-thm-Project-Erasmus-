# Hibrit duplicate development pilotu v1

Tarih: 2026-10-22

Durum: **Development özellik çıkarımı tamamlandı; otomatik karar kuralı/eşikleri henüz
dondurulmadı; holdout görüntüleri açılmadı.**

Bu çalışma bir kanser sınıflandırma sonucu değildir. Amaç, daha sonra aynı kaynak görüntünün
dönüşümlerini tek bir train/validation/test bölümünde tutabilmek için yakın-kopya doğrulayıcısının
sayısal kanıtlarını üretmektir.

## 1. Çalıştırılan komut

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\score_duplicate_development.py
```

Kod: `src/score_duplicate_development.py`

Yerel çıktılar:

- `artifacts/audit/duplicate_hybrid_development_v1/pair_scores.csv`
- `artifacts/audit/duplicate_hybrid_development_v1/report.json`

`artifacts/` Git'e gönderilmez. Bu rapor, doğrulanmış özetin Git ile izlenen kopyasıdır.

## 2. Girdi ve holdout koruması

Ana calibration tablosunda 1.342 çift vardır:

- 671 development çifti skorlandı;
- 671 holdout satırı yalnız sayıldı, holdout görüntü yolları çözülmedi veya görüntüler açılmadı.

Development referansları:

| Referans | Sayı | Kaynak |
|---|---:|---|
| Aynı kaynak | 547 | 520 kontrollü sentetik + 12 SHA-256 exact + 15 AI-destekli görsel aday |
| Bağımsız benzer | 123 | AI-destekli development görsel incelemesi |
| Belirsiz | 1 | `C000620`; eşik metriklerine pozitif/negatif diye katılmadı |
| Toplam | 671 | Yalnız development |

AI-destekli kararlar bağımsız uzman-insan gold standardı değildir. Kontrollü sentetik örnekler de
doğal duplicate yaygınlığını temsil etmez.

## 3. Üretilen kanıtlar

Her çift 256×256 analiz boyutunda, hem orijinal hem yatay çevrilmiş yönelimle incelendi:

1. Var olan 64-bit pHash Hamming mesafesi korundu.
2. ORB yerel anahtar noktaları eşleştirildi.
3. RANSAC, eşleşmelerden tek bir affine dönüşümü destekleyen inlier'ları ayırdı.
4. Phase correlation kaba kaymayı ölçtü.
5. MRI, CT ve Pap-smear çiftlerinde ECC affine hizalama denendi.
6. Hizalanmış geçerli alanda SSIM, NMI, korelasyon ve overlap kaydedildi.

ECC 256 development çiftinde denendi ve 251'inde yakınsadı. Başarısız beş çiftin tamamı görsel
incelemede bağımsız kabul edilmiş Pap-smear çiftleridir. ECC başarısızlığı tek başına negatif karar
olarak kullanılmayacaktır.

## 4. Doğrulama kontrolleri

- 671/671 development çifti işlendi.
- 12 SHA-256 exact pozitifin tamamında ORB inlier oranı, hizalanmış SSIM ve korelasyon 1,0 çıktı.
- Development kontrollü pozitiflerin 270/520'si pHash 12 üzerindedir. Bu, pHash ≤12'nin tek aday
  kapısı olamayacağına ilişkin önceki iki-partition sonucuyla uyumludur.
- Rotation, shift, shear, zoom ve flip pozitiflerinde ORB/RANSAC güçlü kanıt üretti.
- Aşırı karartma (`brightness_0_20`) bazı örneklerde ORB'yi zayıflattı; ORB inlier sayısı iki
  örnekte 0'a kadar indi. Bu nedenle tek ORB eşiği de bütün dönüşümler için yeterli değildir.
- Yeni geometrik çekirdek testleriyle birlikte toplam 23 birim testi Python 3.10'da geçti.

## 5. Doğal development çiftlerindeki ayrım

Yalnız AI-destekli görsel incelemenin kesin pozitif/negatif kararları karşılaştırıldığında:

| Kanıt | 15 aynı-kaynak adayında | 123 bağımsız çiftte |
|---|---:|---:|
| ORB ratio match | 540–2.473 | 4–91 |
| ORB RANSAC inlier | 449–2.360 | 3–17 |
| ORB inlier oranı | 0,662–0,973 | 0,093–0,750 |
| ORB coverage | 0,270–0,717 | 0,000–0,225 |
| En iyi hizalanmış SSIM | 0,777–0,969 | 0,012–0,742 |
| En iyi hizalanmış korelasyon | 0,959–0,999 | -0,299–0,837 |

Development verisinde tek özellikli eşik araması bazı özelliklerde 15 pozitif ve 123 negatifin
tamamını ayırabilmiştir. Örneğin ORB inlier için iki grup arasındaki orta nokta yaklaşık 233,
hizalanmış SSIM için yaklaşık 0,76'dır. **Bunlar aynı development verisinden seçildiği için henüz
başarı sonucu veya dondurulmuş eşik değildir.** Holdout'ta bozulabilirler.

## 6. `C000620` neden belirsiz kaldı?

`C000620` kidney-tumor CT çifti sayısal olarak da iki grubun arasındadır:

| Kanıt | Değer |
|---|---:|
| pHash mesafesi | 8 |
| ORB ratio match | 151 |
| ORB inlier | 44 |
| ORB inlier oranı | 0,291 |
| ORB coverage | 0,095 |
| ECC skoru | 0,960 |
| En iyi hizalanmış SSIM | 0,754 |
| En iyi hizalanmış korelasyon | 0,942 |

ORB inlier sayısı bağımsızların en yükseği olan 17'nin üstünde, doğrulanmış doğal pozitiflerin en
düşüğü olan 449'un çok altındadır. SSIM 0,742 negatif üst sınırı ile 0,777 pozitif alt sınırı
arasındadır. Korelasyon da aynı şekilde iki grup arasına düşer. Bu nedenle mevcut `uncertain`
kararını zorla pozitife veya negatife çevirmemek tutarlıdır.

## 7. Bu pilot neyi kanıtlamaz?

- Development'taki kusursuz ayrım, bağımsız holdout performansı değildir.
- 15 doğal pozitif, bütün veri setindeki dönüşüm çeşitliliğini kanıtlamaz.
- Aynı hasta/seri bilgileri kayıp olduğu için komşu CT/MRI kesitlerini kesin hasta düzeyinde
  çözemez.
- Bu çalışma henüz 130.000 görüntüde aday aramamış, augmentation family üretmemiş veya classifier
  split'i oluşturmamıştır.
- Tek pHash, ORB, ECC, SSIM veya korelasyon değeri nihai karar olmayacaktır.

## 8. Sonraki kontrollü adım

Development kanıtıyla açıklanabilir, çok-kanallı ve gerekirse modaliteye göre değişen bir karar
kuralı tasarlanacaktır. Özellikle:

1. güçlü ORB/RANSAC yolu;
2. ORB'nin karanlık/parlaklık dönüşümünde zayıfladığı çiftler için yoğunluk-yapı yolu;
3. iki yolun arasındaki örnekler için `uncertain` bölgesi;
4. eşiklerin ve parametrelerin değişmez bir JSON yapılandırmasına yazılması

hazırlanacaktır. Bu yapılandırma dondurulup hash'i kaydedilmeden holdout açılmayacaktır. Ayrıca eski
120 çiftlik pHash pilotunun bekleyen etiket tablosu, nihai kalibrasyon kanıtına karıştırılmadan
`legacy pilot` olarak kapatılacak veya açık borç olarak belgede tutulacaktır.
