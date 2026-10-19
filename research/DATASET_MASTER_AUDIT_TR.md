# Multi Cancer Dataset — kapsamlı kaynak, etiket ve kullanım denetimi

Tarih: 2026-10-15  
Durum: **Salt okunur veri denetimi tamamlandı; bu aşamada model eğitilmedi.**

## 1. Kesin özet

- Yerelde kabul edilen **130.000 görüntü**, **8 üst grup** ve **26 alt etikete** ayrılıyor.
- Her alt etikette tam **5.000** kabul edilen görüntü var; bu denge özgün kaynakların doğal
  dağılımı değil, yoğun veri artırma/örnekleme sonucudur.
- Bu koleksiyon tek bir görüntüleme protokolü veya tek bir tanı ekseni değildir. Kan yayması,
  T1 kontrastlı MRI, abdominal CT, Pap smear hücre görüntüsü ve H&E histopatolojiyi karıştırır.
- İzlenebilen özgün kaynak sayıları toplamı yaklaşık **28.360 görüntü/kesit/bölge** eder; bu
  sayı hasta sayısı değildir ve birleşik veri setinde tüm özgün örneklerin aynen korunduğu
  garanti edilemez. 130.000 dosyayı 130.000 bağımsız klinik örnek gibi yorumlayamayız.
- Hasta, slayt, büyütme ve augmentation-aile kimlikleri yeniden adlandırma sırasında büyük ölçüde
  kaybolmuştur. Bu nedenle normal rastgele görüntü bölmesi klinik bağımsızlık sağlamaz.
- Şu anki doğru çalışma hedefi: **26 alt etiketi tahmin etmek**, sonra tahmini etiketten
  **8 üst-grup sonucunu türetmek** ve iki seviyeyi ayrı ayrı ölçmek.
- Veri setinin hiçbir grubunda ortak, güvenilir bir kanser evresi etiketi yoktur.

## 2. Yerel dosya envanteri

| Özellik | Doğrulanan değer |
|---|---:|
| Kabul edilen görüntü | 130.000 |
| Üst grup | 8 |
| Alt etiket | 26 |
| Alt etiket başına görüntü | 5.000 |
| PIL ile açılabilen kabul edilen görüntü | 130.000/130.000 |
| Renk modu | 130.000 RGB |
| Gerçek JPEG içeriği | 125.000 |
| Gerçek PNG içeriği | 5.000 |
| 512×512 | 129.997 |
| 640×640 | 3 |
| Eğitim indeksinden çıkarılan yedek dosya | 2 |
| Kabul edilen dosyaların toplam boyutu | 9.280.007.841 byte |

README bütün dosyaları JPEG ve 512×512 diye tanımlar; dosya başlığı denetimi iki sapma buldu:

1. `breast_malignant` altındaki 5.000 dosyanın adı `.jpg`, fakat gerçek kodlaması PNG'dir.
2. `all_early_0742.jpg`, `all_early_1128.jpg` ve `all_early_3374.jpg` 640×640'tır.

Bu dosyalar bozuk değildir ve eğitim sırasında yeniden boyutlandırılabilir. Fakat format,
sıkıştırma ve yeniden örnekleme farkları modelin tıbbi morfoloji yerine veri hazırlama izlerini
öğrenebileceği anlamına gelir.

## 3. Sekiz üst grubun gerçek yapısı

| Üst grup | Alt sınıf | Yerel görüntü | Görüntü türü | Gerçekte verdiği bilgi |
|---|---:|---:|---|---|
| ALL | 4 | 20.000 | Kan yayması mikroskopisi | Benign referans veya üç B-ALL alt tipi |
| Brain Cancer | 3 | 15.000 | T1 kontrastlı MRI kesiti | Üç beyin tümörü tipi; normal kontrol yok |
| Breast Cancer | 2 | 10.000 | H&E histopatoloji | Benign/malignant |
| Cervical Cancer | 5 | 25.000 | Pap smear hücre mikroskopisi | İki normal, iki abnormal, bir benign hücre tipi |
| Kidney Cancer | 2 | 10.000 | Abdominal CT kesiti | Normal/tumor; malignite ve evre verilmemiş |
| Lung and Colon Cancer | 5 | 25.000 | H&E histopatoloji | İki organ; benign doku ve kanser tipleri |
| Lymphoma | 3 | 15.000 | H&E histopatoloji | Üç malign lenfoid alt tip; normal kontrol yok |
| Oral Cancer | 2 | 10.000 | H&E histopatoloji | Normal oral epitel/OSCC |

Bu tabloya göre üst klasör adı çoğu zaman bir tanı değil, veri kaynağı/organ grubudur.

## 4. Tam 26 etiket sözlüğü

| Yerel etiket | Açık adı | Güvenli yorum | Evre? |
|---|---|---|---|
| `all_benign` | Benign hematogone | Benign referans hücre | Yok |
| `all_early` | Early Pre-B ALL | Malign B-ALL alt tipi | Yok; ilerleme evresi diye sunulmaz |
| `all_pre` | Pre-B ALL | Malign B-ALL alt tipi | Yok |
| `all_pro` | Pro-B ALL | Malign B-ALL alt tipi | Yok |
| `brain_glioma` | Glioma | Üç tümör tipi içinden glioma | Yok |
| `brain_menin` | Meningioma | Üç tümör tipi içinden meningioma | Yok |
| `brain_tumor` | Pituitary tumor | Klasör adı genel görünse de pituitary tumor | Yok |
| `breast_benign` | Benign breast lesion | Benign meme histopatolojisi | Yok |
| `breast_malignant` | Malignant breast lesion | Malign meme histopatolojisi | Yok |
| `cervix_dyk` | Dyskeratotic | Abnormal servikal hücre kategorisi | Yok |
| `cervix_koc` | Koilocytotic | Abnormal servikal hücre kategorisi | Yok |
| `cervix_mep` | Metaplastic | Benign servikal hücre kategorisi | Yok |
| `cervix_pab` | Parabasal | Normal servikal hücre kategorisi | Yok |
| `cervix_sfi` | Superficial-intermediate | Normal servikal hücre kategorisi | Yok |
| `kidney_normal` | Normal kidney | Normal böbrek CT sınıfı | Yok |
| `kidney_tumor` | Kidney tumor | Tümör sınıfı; malignite belirtilmemiş | Yok |
| `colon_aca` | Colon adenocarcinoma | Malign kolon histolojisi | Yok |
| `colon_bnt` | Benign colon tissue | Benign kolon dokusu | Yok |
| `lung_aca` | Lung adenocarcinoma | Malign akciğer histolojisi | Yok |
| `lung_bnt` | Benign lung tissue | Benign akciğer dokusu | Yok |
| `lung_scc` | Lung squamous cell carcinoma | Malign akciğer histolojisi | Yok |
| `lymph_cll` | CLL/SLL | Üç malign lenfoid tipten biri | Yok |
| `lymph_fl` | Follicular lymphoma | Üç malign lenfoma tipinden biri | Yok |
| `lymph_mcl` | Mantle cell lymphoma | Üç malign lenfoma tipinden biri | Yok |
| `oral_normal` | Normal oral epithelium | Normal oral histopatoloji | Yok |
| `oral_scc` | Oral squamous cell carcinoma | Malign OSCC histopatolojisi | Yok |

Makine tarafından kullanılacak aynı eşleme `config/label_taxonomy.json` dosyasındadır.

## 5. Özgün kaynaklar ve birleşik veri setine dönüşüm

### 5.1 ALL

- Kaynak: Mehrad Aria ALL görüntü veri seti.
- 3.256 periferik kan yayması görüntüsü, 89 şüpheli kişi; 25 sağlıklı/benign ve 64 ALL
  hastası.
- Özgün dağılım: benign 504, Early Pre-B 985, Pre-B 963, Pro-B 804.
- 100× mikroskop ve Zeiss kamera; alt tipler flow cytometry ile belirlenmiş.
- Yerelde her biri 5.000'e çıkarılmıştır. Özellikle benign sınıfta yoğun çoğaltma vardır.
- `Early/Pre/Pro`, hastalığın 1–2–3 ilerleme evresi değildir; B-ALL alt tipi adlarıdır.

### 5.2 Brain

- Kaynak: Jun Cheng Figshare brain tumor dataset.
- 233 hastadan 3.064 T1-weighted contrast-enhanced MRI kesiti.
- Özgün dağılım: glioma 1.426, meningioma 708, pituitary tumor 930.
- Normal beyin sınıfı yoktur. Model yalnızca girdi bu üç tümör tipinden biri varsayımıyla
  alt tipi seçebilir.
- Özgün kaynağın hasta kimliği ve tümör maskeleri vardır; birleşik yerel klasörde yalnızca
  yeniden adlandırılmış görüntüler ve sınıf klasörleri kalmıştır. Bu nedenle yerel veriyle
  hasta-bazlı bölme veya gerçek segmentasyon yapılamaz.

### 5.3 Breast

- Bağlantılı Kaggle kaynağı temizlenmiş BreaKHis histopatoloji koleksiyonudur.
- Temizlenmiş sürüm için raporlanan sayı 7.783 görüntüdür: 2.479 benign, 5.304 malignant;
  82 hasta ve farklı mikroskop büyütmeleri bulunur.
- Yerelde iki sınıf da 5.000'dir; hasta ve büyütme bilgisi dosya adlarında korunmamıştır.
- Kritik format sapması: benign dosyalar gerçek JPEG, malign dosyaların tamamı `.jpg` adlı
  gerçek PNG'dir. Bu, benign/malignant ayrımında veri hazırlama kısayolu oluşturabilir.

### 5.4 Cervical

- Kaynak: SIPaKMeD.
- 966 Pap smear cluster görüntüsünden elle kırpılmış 4.049 izole hücre.
- Özgün dağılım: parabasal 787 ve superficial-intermediate 831 normal; metaplastic 793
  benign; dyskeratotic 813 ve koilocytotic 825 abnormal.
- Yerelde her sınıf 5.000'e çıkarılmıştır.
- Bunlar beş serviks kanseri türü veya evresi değildir; hücre morfolojisi kategorileridir.

### 5.5 Kidney

- Kaynak: CT Kidney Normal-Cyst-Tumor-Stone.
- Dhaka'daki hastanelerin PACS sistemlerinden 12.446 CT kesiti: normal 5.077, cyst 3.709,
  tumor 2.283, stone 1.377.
- Hem axial/coronal hem contrast/non-contrast görüntüler vardır.
- Birleşik veri seti cyst ve stone sınıflarını tamamen dışarıda bırakıp yalnızca normal ve
  tumor sınıflarını 5.000'er görüntü yapmıştır.
- `tumor`, tek başına malignite türü veya evre vermez.

### 5.6 Lung and Colon

- Kaynak: LC25000.
- Beş sınıf: benign colon, colon adenocarcinoma, benign lung, lung adenocarcinoma, lung SCC.
- Yayınlanan 25.000 görüntü dengelidir; fakat bunlar sınıf başına yalnızca 250 özgün
  görüntünün augmentation ile 5.000'e çıkarılmasıyla üretilmiştir. Toplam özgün taban 1.250'dir.
- Birleşik yerel sayı da sınıf başına 5.000'dir. Kaynak aile kimliği dosya adlarında yoktur.
- Kesin kopya taramasındaki kopyaların 1.280'i bu beş sınıftadır; ayrıca dönüşmüş yakın
  akrabaların sayısı henüz ölçülmemiştir.

### 5.7 Lymphoma

- Kaynak: IICBU malignant lymphoma koleksiyonu/Kaggle aynası.
- Üç malign tip: CLL, FL ve MCL. Normal/benign kontrol sınıfı yoktur.
- Özgün çalışma 30 vakadan 374 H&E görüntü bölgesi raporlar: CLL 113, FL 139, MCL 122.
- Kaggle aynası 5.400 görüntü ilan eder; birleşik yerel veri 15.000 görüntüdür. Kaynak ile
  yerel veri arasındaki tam çoğaltma zinciri belgelenmemiştir.
- Bu grup en yüksek augmentation/tekrar risklerinden birini taşır.

### 5.8 Oral

- Kaynak: Histopathological Imaging Database for Oral Cancer Analysis.
- 230 hastadan toplam 1.224 H&E görüntüsü: 290 normal ve 934 OSCC.
- 100× grubunda 89 normal/439 OSCC; 400× grubunda 201 normal/495 OSCC vardır.
- Yerelde iki sınıf da 5.000'e çıkarılmıştır; hasta ve 100×/400× bilgileri dosya adlarından
  kaybolmuştur.
- Bu grup gerçek bir normal-versus-OSCC ikili hedefi sağlar; kanser evresi sağlamaz.

## 6. 130.000 sayısının doğru yorumu

| Kaynak grubu | İzlenebilen özgün taban | Yerel toplam | Temel risk |
|---|---:|---:|---|
| ALL | 3.256 | 20.000 | Aynı hücrelerin augmentation akrabaları |
| Brain | 3.064 kesit | 15.000 | Aynı hastadan çok kesit + augmentation |
| Breast | 7.783 temizlenmiş görüntü | 10.000 | Hasta/büyütme kaybı + format kısayolu |
| Cervical | 4.049 hücre | 25.000 | Aynı cluster ve augmentation akrabaları |
| Kidney (seçilen iki sınıf) | 7.360 kesit | 10.000 | Aynı hasta/seri kesitleri + tekrar |
| Lung/Colon | 1.250 özgün görüntü | 25.000 | Yaklaşık 20× augmentation ailesi |
| Lymphoma | 374 özgün bölge | 15.000 | Çok yoğun çoğaltma; yalnız 30 vaka |
| Oral | 1.224 görüntü | 10.000 | Hasta ve büyütme kimliği kaybı |
| **Toplam** | **yaklaşık 28.360** | **130.000** | Bağımsız örnek sayısı çok daha düşük |

Bu toplam, kaynak zincirinin yaklaşık üst görünümüdür; hasta sayısı değildir ve yeniden
derleme sırasında hangi özgün görüntülerin seçildiğini doğrulayan manifest yoktur.

## 7. Kesin kopya denetimi

SHA-256 denetimi yalnız byte-byte aynı dosyaları ölçtü:

| Sonuç | Değer |
|---|---:|
| Kesin kopya grubu | 1.318 |
| Kopya gruplarındaki dosya | 2.673 |
| Fazladan/redundant kesin kopya | 1.355 |
| Alt sınıflar arası aynı dosya grubu | 0 |

Alt sınıf bazında redundant kesin kopya:

| Alt sınıf | Redundant kopya |
|---|---:|
| `lung_scc` | 276 |
| `lung_aca` | 273 |
| `lung_bnt` | 256 |
| `colon_aca` | 247 |
| `colon_bnt` | 228 |
| `kidney_normal` | 75 |

Bu sonuç iyi haber olarak farklı sınıflarda aynı dosya bulunmadığını gösterir. Ancak rotation,
shift, zoom, brightness veya JPEG yeniden kodlama uygulanmış aynı kökenli görüntüler SHA-256 ile
aynı görünmez. Asıl leakage riski için algısal hash/benzerlik gruplama hâlâ gereklidir.

## 8. README'deki augmentation

Birleşik README şu Keras `ImageDataGenerator` ayarlarını bildirir:

- rotation: 10 dereceye kadar;
- width/height shift: %10;
- shear ve zoom: %10;
- horizontal flip;
- brightness: 0.2–1.2;
- fill mode: nearest.

26-sınıf örnek ızgarasında bazı görüntülerin kenarlarında `nearest` dolgusuyla uyumlu uzamış
şeritler görülür. Model bu yapay kenarları öğrenebilir. Ayrıca LC25000 zaten kendi kaynağında
augmentation ile 25.000'e çıkarılmıştır; birleşik README, her kaynak için yeniden üretim
manifesti vermediği için LC sınıflarının ayrıca tekrar artırılıp artırılmadığı kesin değildir.

## 9. Başkaları bu veriyle ne yaptı?

| Çalışma | Gerçek görev | Yaklaşım | Raporlanan/yerel durum | Temel sorun |
|---|---|---|---|---|
| Popular VGG16 notebook | Her üst grup için ayrı 2–5 alt-sınıf modeli | Frozen ImageNet VGG16 | Gruba göre %83.40–%98.77 validation accuracy | Bağımsız test yok; validation'a test denmiş |
| `Cancer classification 97.7 accuracy` | Yalnız ALL'nin 4 sınıfı | 64×64 özel CNN | %97.7 test accuracy, yaklaşık 0.98 macro-F1 reported | Tüm veri seti sonucu değil; augmented image-level split |
| PyTorch ResNet18 notebook | Sekiz ayrı grup içi model | Frozen ImageNet ResNet18 | %84.2–%98.8 reported | Seed yok; olasılık kodu hatalı; grup bazlı görev |
| Dataset-owner ResNet50V2 | Sekiz ayrı grup içi model | Frozen ResNet50V2 | Çıktısız kaynak incelendi | 80/20 train-validation; gerçek test yok |
| RoupB/cancer-app | Tek 26-sınıf modeli ve ayrıca grup modelleri | DenseNet, ResNet, EfficientNet, MobileNet, VGG, ViT | 26 sınıfta %84.90–%94.82 reported; yerel küçük DenseNet pilotu %78.21 | 70/15/15 rastgele görüntü bölmesi; augmentation leakage riski |
| Samayun notebook | 26 sınıf | Birçok CNN, ViT, ensemble, XGBoost; XAI | Kaynakta doğrulanmış çıktı yok | Çok büyük/karmaşık; önce tek modeller doğrulanmalı |

Önemli desen: Popüler çalışmaların çoğu ya sekiz ayrı alt-sınıf modeli kuruyor ya da tek 26-sınıf
modeli kullanıyor. Gerçek “sekiz kanser türü teşhisi” yaptıkları varsayılmamalıdır.

## 10. Bu veriyle yapabileceklerimiz

### Birincil ve savunulabilir görev

Tek görüntüden 26 alt etiketten birini tahmin etmek. Tahmin edilen alt etiketten şu yapı üretilir:

```text
parent_group
leaf_label
anatomical_or_system_group
dataset_status
safe_output
stage_available = false
```

İki seviyede değerlendirme yapılır:

1. 26-leaf accuracy, macro-F1, balanced accuracy, sınıf recall ve confusion matrix.
2. Aynı tahminler 8 parent etikete katlanarak parent accuracy, macro-F1 ve confusion matrix.

Örnek: gerçek `breast_malignant`, tahmin `breast_benign` ise parent sonucu doğru fakat leaf ve
benign/malignant sonucu yanlıştır. Sekiz-sınıf metriğinin sakladığı hata böyle görünür olur.

### Grup içinde anlamlı ikincil ölçümler

- ALL: benign referans versus malignant ALL ve dört alt sınıf.
- Brain: üç tümör tipi arasında sınıflandırma; normal/tumor sonucu yok.
- Breast: benign/malignant.
- Cervix: 5 hücre tipi; ek olarak normal/benign/abnormal katlanmış sonuç düşünülebilir.
- Kidney: normal/tumor; cyst ve stone bu yerel sette yok.
- Lung/Colon: önce gerçek organ leaf adından, sonra benign/malignant/tip.
- Lymphoma: üç malign alt tip.
- Oral: normal/OSCC.

### Bu veriyle yapamayacaklarımız

- bütün organlarda ortak klinik healthy-versus-cancer teşhisi;
- kanser evresi, grade, metastaz, prognoz veya sağkalım tahmini;
- yalnız yerel görüntülerle hasta-bazlı bağımsızlık garantisi;
- bounding-box detection;
- beyin kaynağındaki maskeler yerelde olmadığı için denetimli segmentation;
- klinik kullanım veya yeni hastanelere genelleme iddiası.

Grad-CAM daha sonra modelin baktığı alanı görselleştirebilir; ground-truth tümör maskesi veya
klinik açıklama yerine geçmez.

## 11. Deneyden önce zorunlu veri hazırlığı

1. Kaynak görüntüler değiştirilmeden korunacak.
2. `config/label_taxonomy.json` her örneğin üst/alt ve güvenli anlam alanlarını sağlayacak.
3. Üç 640×640 örnek eğitim loader'ında diğerleriyle aynı giriş boyutuna dönüştürülecek;
   kaynak dosyaya dokunulmayacak.
4. Kesin kopya grubundaki tüm dosyalar aynı train/validation/test bölümünde tutulacak.
5. Perceptual-similarity denetimiyle dönüşmüş augmentation aileleri tahmin edilmeye çalışılacak.
6. Mümkünse aile-duyarlı split ana bilimsel sonuç; sıradan random split yalnız literatür
   karşılaştırması olacak.
7. Dosya formatı, dosya boyutu veya dosya adı modele özellik olarak verilmeyecek.
8. İlk öğrenme kontrolünden ve Mehmet'in açık onayından önce eğitim başlatılmayacak.

## 12. Açık kalan sorular

- Birleşik veri setinin hangi yerel dosyasının hangi özgün hasta/slayt/görüntüden üretildiğini
  gösteren manifest yok.
- Breast büyütme, oral büyütme ve beyin/kidney hasta kimlikleri yeniden adlandırılmış yerel
  dosyalardan geri okunamıyor.
- Lymphoma Kaggle aynasının 5.400 görüntüsü ile özgün 374 bölge ve yerel 15.000 görüntü arasındaki
  tam augmentation zinciri belgelenmemiş.
- Near-duplicate aileleri henüz ölçülmedi; kesin kopya denetimi yalnız ilk katmandır.

Bu bilinmeyenler saklanmayacak; raporda `dataset limitations` olarak açıkça yazılacaktır.

## 13. Kaynak bağlantıları

- Combined Multi Cancer: https://www.kaggle.com/datasets/obulisainaren/multi-cancer
- ALL: https://www.kaggle.com/datasets/mehradaria/leukemia
- Brain: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- Breast: https://www.kaggle.com/datasets/anaselmasry/breast-cancer-dataset
- Cervical/SIPaKMeD: https://www.cs.uoi.gr/~marina/sipakmed.html
- Kidney: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
- LC25000: https://arxiv.org/abs/1912.12142
- Lymphoma: https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification
- Oral: https://doi.org/10.1016/j.dib.2020.105114

