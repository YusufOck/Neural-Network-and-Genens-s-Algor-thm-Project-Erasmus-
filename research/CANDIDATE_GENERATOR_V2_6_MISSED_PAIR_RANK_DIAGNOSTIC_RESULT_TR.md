# Candidate Generator v2.6 — 16 Kaçak Rank Tanısı Sonucu

Tarih: 2026-11-11
İşlem türü: Önceden tanımlanmış, salt-okunur full-index rank diagnostic
Durum: **Tanı tamamlandı; top-K yükseltme hipotezi kapasite ve recall açısından uygun değil**

## 1. Çalışma ve bütünlük

Colab/T4 çalışması frozen commit `c7b06b7c52ba25aba619d1a8d28606f00e6eb9f5` ile return
code 0 verdi. Dataset ZIP'i açılmadı. Mevcut full-retrieval artifact'ları salt-okunur kullanıldı.

Doğrulanan temel durum:

- 130.000 görüntü;
- 1.107 kayıtlı retrieval artifact'ı;
- bu tanıda hash ile doğrulanan 139 artifact;
- 16/16 frozen V2.5 outside positive;
- üç source group: brain MRI, kidney CT ve lymphoma histology;
- bütün gruplarda recomputed robust top-15 = frozen robust top-15;
- config ve analyzer hash'leri predeclared değerlerle eşleşti.
- indirilen gerçek raporu ve bütün yük formüllerini kapsayan 142/142 proje testi geçti.

İndirilen rapor SHA-256:

`88902b9c3810e6f3d945208e96a7d35a3d60a5352eaed98b8dfa60658cd6973f`

## 2. Rank sonucu

| Source group | Hedef çift | Hepsini kapsayan minimum gözlenen K | İlk grid K | Grid içinde tam recovery |
|---|---:|---:|---:|---|
| `brain_ce_mri` | 6 | **2.076** | Yok | Hayır |
| `kidney_ct` | 6 | 28 | 30 | Evet |
| `lymphoma_histology` | 4 | 29 | 30 | Evet |

Brain MRI'daki iki `outside_random` çifti belirleyicidir:

- `V25N000070`: minimum robust K = 140;
- `V25N000107`: minimum robust K = **2.076**.

Rank 2.076, en iyi descriptor kanalı ve en iyi sorgu yönünde bile 2.075 görüntünün hedef eşinden
daha yakın sıralandığı anlamına gelir. Bu çift için K=30, K=100 veya küçük bir top-K artışı yeterli
değildir.

## 3. Kapasite sonucu

V2.5 baseline: 2.994.870 candidate = 23,037 çift/görüntü.

### Yalnız ilgili grup değiştirilirse

| Değişiklik | Full candidate | Çift/görüntü | 25 kapasitesi | Gözlenen hedef recovery |
|---|---:|---:|---|---|
| Brain K=25 | 3.213.258 | 24,717 | Geçer | Hayır |
| Brain K=30 | 3.320.964 | 25,546 | Kalır | Hayır |
| Brain K=100 | 4.709.007 | 36,223 | Kalır | Hayır |
| Kidney K=30 | 3.110.949 | 23,930 | Geçer | Evet |
| Lymphoma K=30 | 3.353.441 | 25,796 | Kalır | Evet |

### Üç gruba ortak K uygulanırsa

| Ortak K | Full candidate | Çift/görüntü | Kapasite | 16/16 recovery |
|---:|---:|---:|---|---|
| 15 | 2.994.870 | 23,037 | Geçer | Hayır |
| 20 | 3.259.424 | 25,072 | Kalır | Hayır |
| 30 | 3.795.614 | 29,197 | Kalır | Hayır |
| 100 | 7.643.907 | 58,799 | Kalır | Hayır |

Ortak K=20 bile kapasiteyi aşarken Brain MRI'nın altı hedefini kapsamaz. Dolayısıyla “üç grupta
K'yı biraz yükselt” yaklaşımı savunulabilir bir V2.6 değildir.

## 4. Yöntemsel yorum

Bu sonuç iki farklı leakage hedefini ayırmamız gerektiğini gösterir:

1. **Aynı 2D kaynak görüntünün augmentation akrabaları:** döndürme, parlaklık, crop ve benzeri
   dönüşümler. Projenin başlangıçtaki temel leakage hedefi budur.
2. **Aynı hasta/scan/seri içindeki farklı MRI veya CT kesitleri:** bunlar görsel olarak ilişkili
   olabilir fakat aynı 2D kaynak görüntü değildir.

İkinci hedef hasta/seri kimliği gerektirir. Yerel dataset bunu taşımadığı için yalnız görsel
yakın-komşulukla eksiksiz hasta/seri grouping garantisi verilemez. Brain MRI'daki K=2.076 bulgusu
bu sınırlamayı doğrudan gösterir.

Frozen V2.5 kararları sonuç görüldükten sonra değiştirilmemelidir; V2.5 başarısız kalır. Ancak yeni
split protokolü hazırlanırken family tanımı sonuç görülmeden açıkça “same-2D augmentation family”
veya “patient/series family” olarak seçilmelidir. Patient/series hedefi seçilirse eksik metadata
nedeniyle tam doğrulama mümkün olmadığı açıkça raporlanmalıdır.

## 5. Hızlandırılmış savunulabilir yol

Teslim süresi nedeniyle aynı top-K aramasını tekrar etmek yerine:

1. family hedefini **same-2D augmentation family** olarak kesin ve yazılı biçimde dondur;
2. MRI/CT adjacent-slice çiftlerini aynı-2D duplicate saymayan yeni karar kılavuzu oluştur;
3. Brain/Kidney/Lymphoma için küçük, yeni ve path-disjoint hedefli audit yap;
4. aynı anda candidate scorer'ı ucuz-önfiltre + pahalı-doğrulama cascade olarak hızlandır;
5. geçen kuraldan family graph ve sabit 70/15/15 split üret;
6. split kesişimi 0 doğrulandıktan sonra baseline model eğitimine başla;
7. hasta/seri kimliği bulunmadığı için residual patient-level leakage dışlanamaz sınırlamasını
   rapora ekle.

Bu rapor V2.6 kuralı seçmez ve scoring/family/split/training yetkisi vermez.
