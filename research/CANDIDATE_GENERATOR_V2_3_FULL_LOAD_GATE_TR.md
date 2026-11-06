# Candidate Generator v2.3 tam-veri yük kapısı sonucu

Tarih: 2026-11-06

Durum: **130.000 görüntünün descriptor, pHash ve source-group içi candidate retrieval
aşamaları tamamlandı ve 1.107 ara artifact hash ile kaydedildi. Nihai candidate listesi
bilinçli güvenlik kapısında durduruldu. Duplicate scoring, family oluşturma,
train/validation/test split ve model eğitimi yapılmadı.**

## 1. Bu çalışma neydi?

Bu çalışma bir kanser sınıflandırma deneyi değildir. Amaç, aynı kaynak görüntüden
üretilmiş olabilecek dosyaları pahalı duplicate doğrulayıcıya göndermeden önce kısa
bir aday listesi oluşturmaktır.

Candidate Generator v2.3 iki arama kanalının birleşimini kullanır:

- **Robust kanal:** ResNet18 descriptor benzerliğine göre her görüntünün dört farklı
  görünümde en yakın 15 komşusunu toplar.
- **pHash kanalı:** Görüntülerin düşük frekanslı genel yapısını karşılaştırır ve
  Hamming uzaklığı 12 veya daha küçük çiftleri toplar.
- **Intersection:** Her iki kanalın da bulduğu çiftlerdir.
- **Union:** En az bir kanalın bulduğu, daha sonra duplicate doğrulayıcıya gönderilmesi
  planlanan benzersiz çiftlerdir.

Burada bir çiftin candidate olması, görüntülerin aynı kaynak olduğuna karar verildiği
anlamına gelmez. Candidate yalnızca "daha ayrıntılı kontrol edilmeli" demektir.

## 2. Kanıt ve bağımsız doğrulama

Colab'da üretilen salt-okunur rapor:

`MyDrive/NNGA_OUTPUTS/candidate_generator_v2_3_full_load_diagnostic_v1.json`

İndirilen yerel dosya Git'e eklenmedi. Yerel SHA-256 değeri:

`be990ff92ec1a07e05b9f3a40cc5e3d16cf04fe712c76aead2258dab55b298cb`

Run-contract SHA-256:

`6c4884d25da3f5d2af2b6d64522b3a78af08db5e05e9ce99e4b13d12686a5261`

Rapor üzerinde aşağıdaki kontroller yerelde yeniden hesaplandı:

| Kontrol | Sonuç |
|---|---:|
| Durum ve salt-okunur işaretleri | Geçti |
| Stage toplamı | 1 + 26 + 26 + 32 + 1.022 = 1.107 |
| Global union formülü | 2.455.082 + 2.345.308 - 178.458 = 4.621.932 |
| Sekiz grubun görüntü toplamı | 130.000 |
| Sekiz grubun robust toplamı | 2.455.082 |
| Sekiz grubun pHash toplamı | 2.345.308 |
| Sekiz grubun intersection toplamı | 178.458 |
| Sekiz grubun union toplamı | 4.621.932 |
| Sekiz grubun pHash block toplamı | 1.022 |
| Her grup için union formülü ve çift/görüntü hesabı | 8/8 geçti |
| Exact-duplicate kapsamı | 1.393/1.393; eksik 0 |
| Seri süre hesabı | 43,5981 gün; raporla aynı |
| İdeal sekiz-worker hesabı | 5,4498 gün; raporla aynı |

Bu kontroller Colab konsol metnini kopyalamak yerine indirilen JSON'un iç
tutarlılığını bağımsız olarak doğrular.

## 3. Tam-veri sonucu

| Ölçüm | Değer |
|---|---:|
| Görüntü | 130.000 |
| Uyumlu kaynak grubu | 8 |
| Robust unique candidate | 2.455.082 |
| pHash unique candidate | 2.345.308 |
| İki kanalda ortak | 178.458 |
| Union candidate | 4.621.932 |
| Union çift/görüntü | 35,5533 |
| Önceden dondurulmuş güvenlik sınırı | 25,0000 |
| Güvenlik kapısı | Geçmedi; işlem doğru biçimde durdu |

Birleşimin neden toplama eşit olmadığı önemlidir: 178.458 çift iki kanal tarafından
da bulunduğu için iki kez sayılmamalıdır. Bu nedenle
`robust + pHash - intersection = union` kullanılır.

## 4. Kaynak grubuna göre yük

| Kaynak grubu | Görüntü | Robust | pHash | Ortak | Union | Çift/görüntü |
|---|---:|---:|---:|---:|---:|---:|
| ALL kan yayması | 20.000 | 420.502 | 14.991 | 12.775 | 422.718 | 21,136 |
| Brain CE-MRI | 15.000 | 272.808 | 1.614.804 | 28.867 | 1.858.745 | 123,916 |
| Breast histology | 10.000 | 277.583 | 578 | 283 | 277.878 | 27,788 |
| Cervix Pap smear | 25.000 | 355.588 | 68.770 | 45.850 | 378.508 | 15,140 |
| Kidney CT | 10.000 | 115.349 | 607.838 | 63.396 | 659.791 | 65,979 |
| Lung/colon histology | 25.000 | 459.991 | 21.349 | 20.188 | 461.152 | 18,446 |
| Lymphoma histology | 15.000 | 334.378 | 10.403 | 5.272 | 339.509 | 22,634 |
| Oral histology | 10.000 | 218.883 | 6.575 | 1.827 | 223.631 | 22,363 |

Brain MRI ve kidney CT açık aykırı gruplardır. Bunun nedeni bu gruplarda çok fazla
görüntünün genel gri-ton yapı bakımından birbirine benzemesi ve pHash'in bunları
yüksek sayıda candidate olarak önermesidir. Bu, hepsinin duplicate olduğu anlamına
gelmez.

## 5. Aşırı yük nereden geliyor?

Sadece pHash'in eklediği çift sayısı:

`2.345.308 - 178.458 = 2.166.850`

Kaynak grubu ayrımı:

| Kaynak grubu | Yalnız pHash'in eklediği çift |
|---|---:|
| Brain CE-MRI | 1.585.937 |
| Kidney CT | 544.442 |
| Diğer altı grubun toplamı | 36.471 |

Brain MRI ve kidney CT birlikte 2.130.379 pHash-only çift üretmiştir. Bu, bütün
pHash-only yükün `%98,3169`'udur. Dolayısıyla problem genel olarak robust kanalın
bozulması değil; global `pHash<=12` kuralının MRI ve CT'ye aynı şekilde uygulanmasıdır.

Yalnızca bir **karşı-olgusal kapasite hesabı** olarak MRI/CT pHash-only çiftleri
çıkarılırsa:

- union: `4.621.932 - 2.130.379 = 2.491.553`
- çift/görüntü: `19,1658`

Bu sayı 25 sınırının altındadır; ancak henüz kabul edilmiş v2.4 kuralı değildir.
Önce MRI/CT'deki gerçek ve kontrollü aynı-kaynak referanslarının bu değişiklikle
kaçırılmadığı gösterilmelidir.

## 6. Exact-duplicate güvenlik kontrolü

Exact-duplicate manifestinde çift kombinasyonu olarak 1.393 bilinen çift vardır.
Candidate union bunların 1.393'ünü de içerir; eksik çift 0'dır. Bu iyi bir pozitif
kontroldür, fakat transformed/near-duplicate recall'unun bütün 130.000 görüntüde
%100 olduğunu tek başına kanıtlamaz.

## 7. Süre ve neden kapı açılmamalı?

Ölçülmüş frozen scorer maliyeti çift başına 0,815 saniyedir. 4.621.932 çift için:

- tek worker seri tahmin: 43,60 gün;
- sekiz worker için matematiksel ideal alt sınır: 5,45 gün.

İkinci sayı gerçek benchmark değildir; I/O, CPU çekişmesi ve paralelleşme kayıpları
nedeniyle gerçek süre daha uzun olabilir. Bu nedenle yalnızca kapıyı 25'ten yukarı
çekip 4,62 milyon çifti skorlamak bilimsel ve operasyonel olarak savunulamaz.

## 8. Verilen karar

- V2.3 artifact'ları korunacak ve değiştirilmeyecek.
- 25 çift/görüntü güvenlik kapısı yükseltilmeyecek.
- 4.621.932 çift duplicate scorer'a gönderilmeyecek.
- Henüz family oluşturulmayacak.
- Henüz train/validation/test split hazırlanmayacak.
- Henüz model eğitilmeyecek.

Bu sonuç başarısız bir Colab çalışması değildir. Tam tersine, önceden koyulan
güvenlik mekanizması küçük pilot tahmininin tam veriye genellenmediğini yakalamıştır.

## 9. Sıradaki kontrollü adım

Bir sonraki çalışma bir **modality-aware v2.4 karşı-olgusal candidate deneyi**
olacaktır. Yeni bir kuralı doğrudan uygulamak yerine şu hipotezler mevcut 1.107
artifact üzerinden salt-okunur karşılaştırılacaktır:

1. Robust top-15 bütün kaynak gruplarında korunur.
2. pHash<=12 histoloji, kan yayması ve Pap smear gruplarında korunur.
3. Brain MRI ve kidney CT'de pHash kanalı kaldırılır veya ayrı, kanıtla seçilen bir
   eşiğe indirilir.
4. Her varyant exact 1.393 çift, kontrollü pozitifler ve 23 doğal aynı-kaynak
   referansı üzerinde recall kaybı açısından kontrol edilir.
5. Kabul için hem referans recall kapıları hem de en fazla 25 çift/görüntü yük kapısı
   birlikte geçmelidir.

Bu hipotez açıklanıp Mehmet tarafından onaylanmadan v2.4 kodu veya yeni candidate
çıktısı üretilmeyecektir.
