# Dondurulmus duplicate karar kurali v1

Tarih: 2026-10-25

Durum: **Kural yalniz development kanitiyla donduruldu. Holdout satiri veya goruntusu acilmadi.**

Bu calisma kanser siniflandirma sonucu degildir. Amac, ileride ayni kaynak goruntunun farkli
versiyonlarini train/validation/test arasinda bolmemek icin skorlanmis goruntu ciftlerini
`same_source`, `independent` veya `uncertain` olarak siniflandirmaktir.

## 1. Neden uc karar var?

- `same_source`: Ayni kaynak goruntuye iliskin yeterince guclu ve birlikte tutarli kanit var.
- `independent`: Geometrik ve yapisal kanitlar birlikte dusuk.
- `uncertain`: Kanitlar iki karar icin de yeterli degil; zorla karar verilmez ve insan incelemesi
  gerekir.

Yanlis bir `independent` karari ayni ailenin split'ler arasinda bolunmesine ve data leakage'e yol
acabilecegi icin, varsayilan karar `uncertain` olarak tasarlanmistir.

## 2. Dondurulan karar sirasi

Kural dosyasi: `config/duplicate_rule_v1.json`

SHA-256:

```text
5ffa80daa6e6ab0eb5f3ac179073de2f0f018325310310b33adc4c137710b645
```

Karar yollarinin onceligi:

1. SHA-256 birebir esitlik;
2. guclu ORB/RANSAC geometrisi;
3. ORB'nin asiri parlaklik degisiminde zayifladigi ciftler icin guclu yapisal/yogunluk kaniti;
4. konservatif bagimsiz karari;
5. hicbiri saglanmazsa `uncertain`.

### Guclu geometri yolu

Asagidaki kosullarin hepsi gerekir:

- affine donusum makul;
- ORB/RANSAC inlier sayisi en az 30;
- inlier orani en az 0,55;
- yerel nokta kapsami en az 0,20.

### Guclu yapisal/yogunluk yolu

Asagidaki kosullarin hepsi gerekir:

- hizalanmis SSIM en az 0,80;
- hizalanmis korelasyon en az 0,95;
- hizalanmis NMI en az 1,25.

Bu yol, `brightness_0_20` gibi ORB anahtar noktalarini azaltan kontrollu pozitifleri korur.

### Konservatif bagimsiz yolu

Asagidaki kosullarin hepsi gerekir:

- ORB/RANSAC inlier sayisi en fazla 20;
- hizalanmis SSIM en fazla 0,75;
- hizalanmis korelasyon en fazla 0,85.

pHash son karari vermez; yalniz aday uretiminde kullanilir. ECC ve phase correlation hizalama
kanitidir; tek basina `same_source` karari vermez. C000620'nin ECC skoru yuksek olmasina ragmen
kararin `uncertain` kalmasi bu korumanin bilincli sonucudur.

## 3. Development uygulamasi

Komut:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\evaluate_duplicate_rule_development.py
```

Sonuc:

| Referans | Karar | Sayi |
|---|---|---:|
| same_source | same_source | 547 |
| independent | independent | 123 |
| uncertain | uncertain | 1 |

Karar yollarinin dagilimi:

| Yol | Sayi |
|---|---:|
| SHA-256 exact | 12 |
| Guclu ORB/RANSAC geometrisi | 522 |
| Guclu yapisal/yogunluk kaniti | 13 |
| Konservatif independent | 123 |
| Uncertain | 1 |

Kan yaymasi, MRI, H&E histopatoloji, Pap smear ve CT development orneklerinin tamaminda referans
kararlar korundu. C000620 yine `uncertain` kaldi.

## 4. Sonucun dogru yorumu

Development uzerindeki bu ayrim **yuzde 100 gercek test basarisi degildir**. Esikler ayni
development verisinin dagilimlari gorulerek secilmistir. Bu sonuc yalnizca kuralin tasarlandigi
kanitla tutarli oldugunu ve kodun beklenen kararlari tekrar uretebildigini gosterir.

Gercek bagimsiz sinav, bu JSON ve SHA-256 degistirilmeden holdout uzerinde yapilacaktir. Holdout
sonucu goruldukten sonra esikler degistirilirse o holdout artik test degil development olur ve yeni
bir kapali orneklem gerekir.

## 5. Teknik dogrulama

- Python 3.10 testleri: 31/31 gecti.
- Development girdisinin SHA-256 degeri dondurulan kanitla eslesiyor.
- Kural JSON'u bir karakter bile degisirse SHA-256 yan dosyasi dogrulamayi durduruyor.
- Development degerlendiricisi holdout satiri iceren CSV'yi reddediyor.
- Eksik sayisal kanit `independent` degil `uncertain` uretiyor.
- Hicbir goruntu ailesi veya classifier split'i olusturulmadi.

## 6. Sonraki kontrollu adim

Holdout acilmadan once kural dosyasi, hash dosyasi, kod, test ve bu rapor Git ile kaydedilecektir.
Daha sonra holdout review adaylari algoritma kararlarini gormeden etiketlenecek; ayni dondurulmus
kural holdout skorlarina bir kez uygulanacak ve sonuclar modalite bazinda raporlanacaktir.
