# RoupB/cancer-app Ayrıntılı Kod ve Yöntem Denetimi

Denetim tarihi: 2026-10-15  
Kaynak depo: https://github.com/RoupB/cancer-app  
İncelenen commit: `24cb2d3d1282796b0e88fdce3b1cb6fc3ce29c50`  
Yerel kopya: `research/external/cancer-app`  
Durum: Kaynak kod, notebook hücreleri ve kayıtlı çıktılar statik olarak incelendi. Bu denetim sırasında üçüncü taraf kodu veya model eğitimi çalıştırılmadı.

## Kısa hüküm

RoupB, şimdiye kadar bulduğumuz çalışmalar arasında bizim planladığımız **26 alt sınıflı tek-model sınıflandırmasına en yakın dış çalışmadır**. Depodaki `All Images.ipynb` defterleri 26 yaprak etiketi tek sınıflandırıcıda kullanır. Ayrıca her organ/veri grubu için ayrı modeller, Streamlit arayüzü, ilk üç tahmin, Grad-CAM, LIME ve McNemar testi içerir.

Fakat bu depo bizim projemizin doğrudan teslim edilebilir çözümü değildir:

- Sekiz üst grup ayrıca ölçülmüyor.
- Genetik algoritma yok.
- Rastgele görüntü düzeyinde bölme, önceden artırılmış ve tekrarlı görüntüleri eğitim ile test arasında ayırabiliyor.
- Çoğu deney yalnızca accuracy veriyor; macro-F1, sınıf başına recall ve karışıklık matrisi yok.
- Modellerin karşılaştırma koşulları eşit değil.
- İstatistik defterinin veri boyutu ve sonucu, 26-sınıflı notebook sonuçlarıyla tutarsız.
- Uygulamadaki bazı tıbbi açıklamalar kaynak etiketlerinin gerçek anlamıyla çelişiyor.

Sonuç: **Kodu körlemesine kopyalamayacağız.** RoupB’yi 26-sınıflı karşılaştırma/reprodüksiyon referansı olarak tutacağız; veri bölme, değerlendirme ve etiket açıklamalarını kendi denetlenmiş boru hattımızda düzelteceğiz.

## 1. Depoda gerçekte ne var?

Depoda 27 model notebook’u, bir Streamlit uygulaması ve bir istatistik notebook’u bulunuyor.

### Tek 26-sınıflı modeller

Yedi `All Images` notebook’u aynı 26 klasör adını hedef sınıf olarak kullanıyor:

- DenseNet121
- EfficientNet-B0
- MobileNetV2
- MobileNetV3-Small
- ResNet18
- VGG16
- ViT-B/16

### Grup içi modeller

DenseNet121 için bütün gruplara ait ayrı notebook’lar var: ALL, Brain, Breast, Cervix, Colon, Kidney, Lung, Lymph ve Oral. Bazı gruplar ayrıca EfficientNet-B0, MobileNetV3, ResNet18 veya ViT ile denenmiş.

Bu iki yaklaşım arasındaki fark:

- `All Images`: Görüntüden doğrudan 26 etiketten birini seçiyor.
- Grup içi model: Kullanıcı önce organ/veri grubunu biliyor veya seçiyor; model yalnızca o grubun 2-5 alt sınıfından birini seçiyor.

Bizim planladığımız ana yön ilkine yakındır: 26 yaprak etiketi tahmin et, sonra o etiketten sekiz üst grubu türet ve iki seviyeyi de ölç.

## 2. `All Images` veri ve eğitim akışı

CNN notebook’larının ortak işlemi şöyledir:

1. Sekiz ana klasörün altındaki 26 alt klasörü dolaşır.
2. Her alt sınıftan en fazla 5.000 dosyayı `random.shuffle` ile karıştırır.
3. Dosyaları fiziksel olarak `%70 train / %15 validation / %15 test` klasörlerine kopyalar.
4. Görüntüleri 128×128 boyutuna getirir.
5. Piksel kanallarını `[0.5, 0.5, 0.5]` ortalama ve standart sapmayla normalize eder.
6. ImageNet ağırlıklı omurgayı dondurur ve yalnızca son sınıflandırıcı katmanı 10 epoch eğitir.
7. Cross-entropy loss ve Adam, `lr=0.001` kullanır.
8. Validation ve test için yalnızca genel accuracy hesaplar.

CNN’lerde sınıf başına 5.000 görüntü olduğu için toplam 130.000 görüntü kullanılır:

- train: 91.000
- validation: 19.500
- test: 19.500

ViT notebook’u farklıdır: sınıf başına yalnızca 500 görüntü, 224×224 giriş ve batch size 8 kullanır. Toplam 13.000 görüntünün yaklaşık 9.100/1.950/1.950 bölünmesiyle çalışır.

## 3. Notebook’larda kayıtlı 26-sınıf sonuçları

Bu değerler RoupB notebook çıktılarında **reported** durumundadır; bizim tam veride yeniden ürettiğimiz sonuçlar değildir.

| Model | Sınıf başına görüntü | Giriş | Batch | Test accuracy | Adil kıyas notu |
|---|---:|---:|---:|---:|---|
| MobileNetV3-Small | 5.000 | 128 | 16 | 84,90% | Diğer CNN’lerle ortak split klasörü |
| ResNet18 | 5.000 | 128 | 16 | 88,02% | Yalnızca accuracy |
| VGG16 | 5.000 | 128 | 16 | 88,44% | Yalnızca accuracy |
| EfficientNet-B0 | 5.000 | 128 | 16 | 89,08% | Yalnızca accuracy |
| MobileNetV2 | 5.000 | 128 | 16 | 90,46% | Yalnızca accuracy |
| DenseNet121 | 5.000 | 128 | 16 | 92,21% | Son epoch checkpoint’i |
| ViT-B/16 | **500** | **224** | **8** | 94,82% | Farklı veri miktarı, giriş ve split klasörü |

ViT sonucu aynı deney bütçesiyle üretilmediği için bu tablo “ViT kesin en iyi modeldir” sonucunu desteklemez. Ayrıca hiçbir `All Images` notebook’u macro-F1 veya 26 sınıfa ait recall değerlerini vermiyor.

## 4. Grup içi DenseNet121 sonuçları

Aşağıdaki kayıtlı test accuracy değerleri de yalnızca **reported** sonuçlardır:

| Ayrı model | Sınıf sayısı | Test accuracy |
|---|---:|---:|
| ALL | 4 | 99,43% |
| Brain | 3 | 92,22% |
| Breast | 2 | 90,73% |
| Cervix | 5 | 95,01% |
| Colon | 2 | 98,40% |
| Kidney | 2 | 93,67% |
| Lung | 3 | 95,78% |
| Lymph | 3 | 84,84% |
| Oral | 2 | 80,20% |

Bu değerler organın önceden bilindiği daha dar görevlerdir. Tek 26-sınıflı modelin sonucuyla aynı problem değildir. Örneğin oral model yalnızca `oral_normal` ile `oral_scc` arasında seçim yapar; görüntünün oral doku olup olmadığını keşfetmez.

## 5. Veri bölme sorunu ve ölçülen sızıntı riski

RoupB bölmeyi veri setinin daha önce artırılmış 5.000’lik alt sınıf klasörleri üzerinde yapıyor. Dosya ailesi, hasta, slayt veya orijinal görüntü kimliğiyle gruplama yok.

Aynı algoritma (`seed=42`, 70/15/15) bizim yerel veri kopyamıza salt-okunur biçimde simüle edildi. Daha önce hesaplanan SHA-256 grupları kullanıldığında:

- 1.318 byte-byte aynı görüntü grubunun **615’i birden fazla split’e dağılıyor**.
- Bu çapraz-split gruplarında 1.252 dosya bulunuyor.
- 285 grup train+test, 279 grup train+validation, 49 grup validation+test ve 2 grup üç split arasında dağılıyor.
- Etkilenen sınıflar: `lung_aca` 118 grup, `colon_aca` 111, `colon_bnt` 104, `lung_bnt` 114, `lung_scc` 128 ve `kidney_normal` 40.
- Kod dosya türünü ve yedek adlarını filtrelemiyor. Yerel simülasyonda iki `*.bak.jpg` dosyası seçildi; 5.000 dosya sınırı yüzünden iki normal dosya dışarıda kaldı.

Bu sayı RoupB’nin bilgisayarındaki dosya listeleme sırasının birebir kanıtı değildir; **onun bölme kodunu bizim aynı veri setimize uyguladığımızda oluşan doğrudan sonuçtur**. Yine de algoritmanın aynı görüntüyü train ve testte tutmayı önlemediğini kesin olarak gösterir. Görsel olarak benzer ama byte olarak farklı artırılmış akrabalar bu sayıya dahil değildir; gerçek sızıntı riski daha büyük olabilir.

## 6. Ön işleme ve transfer learning sorunları

### ImageNet normalizasyonu

DenseNet, ResNet, EfficientNet, MobileNet ve VGG ImageNet ağırlıklarıyla açılıyor; fakat CNN notebook’ları ImageNet’in beklediği kanal ortalama/standart sapmasını değil `[0.5]/[0.5]` kullanıyor. Bu mutlaka modeli bozduğu anlamına gelmez, ancak ön-eğitim dağılımıyla uyumu azaltır ve doğru bir karşılaştırmada düzeltilmelidir. ViT ise ImageNet ortalama ve standart sapmasını kullanıyor; bu da ViT karşılaştırmasını daha da eşitsiz yapıyor.

### Donmuş omurga ayrıntısı

Parametrelerin `requires_grad=False` yapılması ağırlık gradyanlarını durduruyor. Ancak eğitim sırasında bütün model `model.train()` moduna geçiriliyor. BatchNorm katmanlarının çalışan ortalama/varyans değerleri böylece değişebilir. “Omurga tamamen sabit” iddiası teknik olarak tam değildir. Düzeltilmiş denemede donmuş omurganın değerlendirme modu veya açıkça seçilmiş fine-tuning politikası kontrol edilmelidir.

### Model seçimi yok

Epoch boyunca validation loss/accuracy izlenmiyor, early stopping uygulanmıyor ve “en iyi validation checkpoint’i” seçilmiyor. DenseNet son epoch sonunda kaydediliyor. Dolayısıyla son modelin gerçekten en iyi genelleştiren epoch olduğu gösterilmiyor.

### Ek eğitim artırması yok

`All Images` CNN akışında `RandomHorizontalFlip`, `RandomRotation` veya benzeri train-only augmentation yok. Fakat kaynak veri zaten artırılmış olduğundan, asıl öncelik yeni augmentation eklemekten önce artırılmış akrabaları split’ler arasında ayırmaktır.

## 7. Değerlendirme ve istatistik defterindeki kritik tutarsızlık

`statistical_test.ipynb` dört model için tek bir `y_true` ve tahmin dosyası indiriyor. Her dizinin uzunluğu **2.250**. Oysa:

- 26-sınıflı CNN test kümesi 19.500 görüntü olmalıdır.
- 26-sınıflı ViT test kümesi 1.950 görüntüdür.
- 2.250, üç adet 5.000’lik sınıfın %15 testine karşılık gelir ve bir organ-içi deneye benzemektedir.

Bu nedenle istatistik defteri 26-sınıflı `All Images` modellerinin adil karşılaştırmasını kanıtlamaz.

Daha ciddi olarak DenseNet–EfficientNet McNemar tablosu şunu içeriyor:

- DenseNet doğru: `(96 + 633) / 2250 = 32,40%`
- EfficientNet doğru: `(1371 + 633) / 2250 = 89,07%`

Notebook buna rağmen “DenseNet performs the best” sonucunu yazıyor. P-değerinin küçük olması yalnızca iki modelin hata davranışının farklı olduğunu söyler; hangi modelin daha iyi olduğunu tek başına söylemez. Bu tabloda yön EfficientNet lehinedir. DenseNet tahmin dosyası yanlış sınıf eşlemesi, yanlış deney veya yanlış örnek sırası içeriyor olabilir.

Ek sorunlar:

- Kod yalnızca dizi uzunluklarını eşitliyor; tahminlerin aynı dosya sırasına ait olduğunu dosya kimlikleriyle doğrulamıyor.
- Altı ikili test için çoklu karşılaştırma düzeltmesi yok. Basit Bonferroni sınırı `0.05/6 = 0.0083` olsaydı EfficientNet–MobileNet `p=0.0257` anlamlı sayılmazdı.
- Tahmin dosyaları `torch.load(..., weights_only=False)` ile yükleniyor. Bu pickle tabanlı dosyalar güvenilmeyen kaynaktan doğrudan çalıştırılmamalıdır.

Bu sebeple depodaki McNemar sonucu projemizde delil olarak kullanılmayacak; gerekirse aynı sabit test manifesti ve dosya kimlikleriyle sıfırdan üretilecektir.

## 8. Streamlit uygulamasından alınabilecek ve alınamayacak fikirler

Uygulama iki kullanım sunuyor:

- `Any`: 26-sınıflı DenseNet modeli.
- Kullanıcının seçtiği organ/grup: o gruba özel DenseNet modeli.

Faydalı yazılım fikirleri:

- Checkpoint içinde `class_to_idx`, görüntü boyutu ve sınıf sayısını saklamak.
- İlk üç tahmini göstermek.
- Eğitilmiş model üzerinde Grad-CAM/LIME görselleştirmesi sunmak.
- Son aşamada basit bir Streamlit demonstrasyonu yapmak.

Olduğu gibi kullanılmaması gereken noktalar:

- Başlık ve metinler sistemi “AI Cancer Diagnosis” diye sunuyor; veri klinik tanı doğrulaması sağlamıyor.
- README böbrek verisini MRI diye yazıyor; kaynak ve yerel taksonomi bunun abdominal CT olduğunu gösteriyor.
- `all_early`, `all_pre`, `all_pro` açıklamaları hastalık evresi gibi yazılmış. Bunlar Early Pre-B, Pre-B ve Pro-B ALL alt tipleridir; ilerleme evresi değildir.
- `cervix_mep` “precancerous” diye sunuluyor; SIPaKMeD taksonomisinde metaplastic sınıf benign hücre kategorisidir.
- Brain ve lymphoma gruplarında normal kontrol olmadığı için bu modeller “kanser var/yok” testi değildir.
- Olasılık yüzdeleri kalibre edilmemiş softmax değerleridir; klinik güven olasılığı diye yorumlanamaz.

Bizim arayüzümüz olursa metinleri `config/label_taxonomy.json` içindeki güvenli açıklamalardan almalıdır.

## 9. Biz neyi kullanacağız?

### Alınabilecek iskelet fikirleri

1. Tek 26-sınıflı sınıflandırıcı fikri.
2. DenseNet121/ResNet18 gibi ön-eğitimli modellerle transfer learning başlangıcı.
3. Model checkpoint’inde sınıf eşlemesi ve ön işleme metadatası saklama.
4. Aynı test örnekleri üzerinde birden fazla modeli karşılaştırma fikri.
5. Model doğrulandıktan sonra Grad-CAM ve basit demo arayüzü.

### Düzelterek yeniden kuracağımız parçalar

1. Fiziksel dosya kopyalamak yerine sabit bir split manifesti.
2. Byte-identical grupları aynı split’te tutma; mümkün olursa perceptual aile denetimi.
3. Tüm modeller için aynı örnekler, aynı çözünürlük veya açıkça gerekçelendirilmiş adil protokol.
4. Accuracy yanında macro-F1, balanced accuracy, sınıf başına precision/recall/F1 ve confusion matrix.
5. 26 yaprak sonucu yanında yapraktan türetilmiş sekiz üst-grup metriği.
6. ImageNet ağırlıklarıyla uyumlu ön işleme.
7. En iyi validation checkpoint’i ve tekrarlanabilir seed/ortam kaydı.
8. Kaynak-destekli etiket açıklamaları; stage ve evrensel kanser-var/yok iddiası yok.
9. İstatistik karşılaştırmasında dosya kimliğiyle eşleşmiş ortak test seti.

### Kullanmayacağımız parçalar

1. RoupB’nin accuracy değerlerini bizim sonucumuz gibi yazmak.
2. ViT’yi farklı veri bütçesiyle “kazanan model” ilan etmek.
3. Mevcut `data_split` algoritmasını ana bilimsel split yapmak.
4. Mevcut McNemar CSV’sini model üstünlüğü kanıtı saymak.
5. Uygulamadaki hatalı tıbbi açıklamaları kopyalamak.

## 10. Kontrollü yeniden üretim planı

RoupB unutulmayacak; dış referans ve kontrollü baseline olarak saklanacak. Fakat sonraki kod çalıştırma, Mehmet’in adımı öğrenip açıkça onaylamasından sonra yapılacak.

Önerilen sıra:

1. **Ders:** 70/15/15 split, data leakage, transfer learning, accuracy ve macro-F1 kavramlarını küçük örnekle anlamak.
2. **Split deneyi:** Bizim sabit ve duplicate-aware 26-sınıf manifestimizi oluşturup beraber kontrol etmek.
3. **RoupB’ye yakın baseline:** DenseNet121, 128 px, donmuş omurga, 26 sınıf. RoupB ile karşılaştırılabilir ayarları açıkça işaretlemek.
4. **Düzeltilmiş baseline:** Aynı sabit split, doğru normalizasyon, best-checkpoint ve tam metrikler.
5. **İki seviye değerlendirme:** 26-yaprak ve türetilmiş 8-üst-grup sonuçlarını aynı tahminlerden hesaplamak.
6. **Model karşılaştırması:** Aynı split ve bütçeyle ResNet18/DenseNet121/EfficientNet-B0.
7. **GA:** Kararlı baseline sonrasında sınırlı hiperparametre araması; eşit bütçeli random search ile kıyas.
8. **XAI ve demo:** Son seçilen model için doğru/yanlış örneklerde Grad-CAM; en son Streamlit arayüzü.

## 11. Kanıt durumu

- `reported`: Bu belgedeki RoupB accuracy değerleri ve notebook çıktıları.
- `reproduced`: Daha önce yerelde yapılan küçük DenseNet121 pilotu: 26 sınıf, sınıf başına 100 görüntü, 2 epoch, test accuracy 78,21% ve macro-F1 76,18%. Bu tam 92,21% iddiasının reprodüksiyonu değildir.
- `verified audit`: Depo yapısı, kod parametreleri, notebook kayıtlı çıktıları, commit kimliği, istatistik tablosu aritmetiği ve yerel veri üzerindeki exact-duplicate split simülasyonu.
- `planned`: Duplicate-aware ana split, tam ölçekli adil model karşılaştırması, sekiz üst grup metriği, GA ve XAI.

