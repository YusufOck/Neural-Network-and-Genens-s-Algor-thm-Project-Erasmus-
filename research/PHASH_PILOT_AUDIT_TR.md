# pHash pilotu: ayrıntılı sonuç ve görsel denetim

Tarih: 2026-10-19

Bu belge `src/audit_perceptual_duplicates.py` ile Mehmet tarafından çalıştırılan ilk pHash
pilotunun inceleme kaydıdır. Bu çalışma model eğitimi, train/validation/test ayrımı veya nihai
eşik seçimi değildir. Amaç, aynı kaynak görüntüden üretilmiş olabilecek dosyaları bulmak için
pHash'in bu veri setinde güvenilir bir **aday üreticisi** olup olmadığını sınamaktır.

## 1. Çalıştırmanın kapsamı

Komut iki farklı görüntü türünü temsil eden alt sınıfı taradı:

- `brain_glioma`: 5.000 beyin MR görüntüsü;
- `lung_scc`: 5.000 akciğer histopatoloji görüntüsü.

Her alt sınıfta bütün ikililer karşılaştırıldı. 5.000 görüntünün ikili kombinasyon sayısı
`5.000 × 4.999 / 2 = 12.497.500` olduğundan toplam 24.995.000 çift değerlendirilmiştir. Her
uzaklık bandından sabit seed ile 12 çift seçilmiş; 2 alt sınıf × 5 bant × 12 çift = 120 çift,
20 temas sayfasında görsel olarak incelenmiştir.

Çıktıdaki `pHash: 10000/10000` bütün seçili görüntülerin hash'inin üretildiğini; iki ayrı
`4999/4999` satırı ise her sınıftaki 5.000 görüntünün tüm benzersiz çiftlerinin tarandığını
gösterir. Çalışma yarıda kesilmemiştir ve raporda okuma/hata uyarısı yoktur.

## 2. pHash uzaklığı burada ne demektir?

Her görüntü gri tona çevrilir, 32×32'ye küçültülür, DCT ile genel görüntü yapısı frekanslara
ayrılır ve düşük frekanslardan 64 bitlik bir imza üretilir. İki imzanın farklı bit sayısı Hamming
uzaklığıdır:

- `0`: pHash imzaları aynı;
- küçük değer: genel görsel yapı çok benzer;
- büyük değer: genel yapı giderek daha farklı.

Bu, dosyaların aynı olduğunu kanıtlamaz. SHA-256 byte-byte kimliği kanıtlarken pHash yalnızca
görsel benzerlik adayı üretir. İki farklı MR kesiti aynı siyah arka plana ve kafa geometrisine
sahip olduğu için küçük pHash uzaklığı alabilir. Tersine, ağır kırpılmış/döndürülmüş gerçek bir
akraba daha büyük uzaklık alabilir.

Bu uygulamadaki median tabanlı 64-bit imzalarda 1-bit sayısı hemen hemen sabit kaldığı için
gözlenen Hamming uzaklıkları çifttir: 0, 2, 4, 6, ... Bu nedenle `03-04`, `05-06` ve `07-08`
bantlarında fiilen yalnızca 4, 6 ve 8 görülmesi hata değildir. Bant adları genel aralık olarak
yazılmıştır; sonraki sürümde yalnızca gerçek değerleri göstermek daha anlaşılır olacaktır.

## 3. Sayısal sonuçlar

| Alt sınıf | 0–2 | 3–4 | 5–6 | 7–8 | 9–12 | Toplam ≤12 | Tüm çiftlere oran |
|---|---:|---:|---:|---:|---:|---:|---:|
| brain_glioma | 137 | 1.217 | 5.513 | 18.647 | 148.478 | 173.992 | %1,39221 |
| lung_scc | 389 | 207 | 269 | 262 | 604 | 1.731 | %0,01385 |

Beyin grubunda 12 veya daha küçük uzaklıklı aday sayısı akciğer grubunun yaklaşık 100,52
katıdır. Bu, beyin grubunda 100 kat fazla gerçek kopya bulunduğu anlamına gelmez. Görsel denetim,
ana nedenin MR görüntülerindeki ortak siyah arka plan, kafa çerçevesi ve merkezi anatomi olduğunu
göstermiştir. Yani pHash bu ortak düzeni “aynı kaynak” sinyaliyle karıştırmaktadır.

### SHA-256 pozitif kontrolü

`lung_scc` içinde önceki SHA-256 denetiminden bilinen 273 birebir kopya grubu ve bu gruplardan
oluşan 279 birebir-kopya çifti vardır. 279 çiftin tamamı pHash uzaklığı 0 vermiş, hiçbiri sıfırdan
büyük uzaklığa düşmemiştir. Bu önemli bir sağlamlık kontrolüdür: pHash hesaplaması bilinen tam
kopyaları kaçırmamıştır.

`brain_glioma` içinde SHA-256 ile bilinen birebir kopya grubu yoktur; bu yüzden aynı pozitif
kontrol o sınıfta yapılamamıştır.

### Benzersiz pHash sayıları

- `brain_glioma`: 5.000 dosyada 4.999 benzersiz pHash; iki dosyalı tek hash çakışma grubu.
- `lung_scc`: 5.000 dosyada 4.716 benzersiz pHash; 281 çakışma grubu ve bu gruplarda 565 dosya.
  Grupların 278'i iki, 3'ü üç dosyalıdır.

Bir pHash çakışması “dosyalar kesin aynıdır” demek değildir. Sadece 64-bit görsel özetlerinin
aynı olduğunu söyler; bu yüzden SHA-256 ve görsel/geometrik kontrolle birlikte yorumlanmalıdır.

## 4. 120 örneğin görsel denetimi

Temas sayfalarındaki her satır bir çift gösterir. Sol ve sağ görsel aynı sahne/aynı anatomik
kesit olup yalnızca parlaklık, renk, yeniden kodlama, küçük kayma veya başka bir dönüşümle
değişmişse `aynı-kaynak adayı`; yalnızca genel görünüşü benziyorsa `bağımsız/yanlış pHash adayı`
olarak yorumlanmıştır. Hasta/slide kimliği bulunmadığından “aynı hasta” iddiası yapılmamıştır.

| Alt sınıf ve bant | İncelenen | Güçlü aynı-kaynak adayı | Bağımsız/yanlış aday | Görsel yorum |
|---|---:|---:|---:|---|
| brain_glioma 0–2 | 12 | 4 | 8 | Çok küçük uzaklıkta bile ortak MR düzeni ciddi false positive üretiyor. |
| brain_glioma 3–4 | 12 | 1 | 11 | Eşik 4 yapılırsa bağımsız görüntülerin çoğu yanlış birleşir. |
| brain_glioma 5–6 | 12 | 0 | 12 | Örneklerin tamamı yalnızca genel yapı bakımından benzer. |
| brain_glioma 7–8 | 12 | 0 | 12 | Aynı kaynak olduğuna dair görsel kanıt yok. |
| brain_glioma 9–12 | 12 | 0 | 12 | Aday hacmi çok büyük, örneklerin tamamı false positive görünüyor. |
| lung_scc 0–2 | 12 | 12 | 0 | Aynı mikroskobik alan; aralarında birebir ve renk/parlaklık türevleri var. |
| lung_scc 3–4 | 12 | 12 | 0 | Alan yapısı ve hücre dizilimi eşleşiyor. |
| lung_scc 5–6 | 12 | 12 | 0 | Renk/geometri değişse de aynı içerik açıkça korunuyor. |
| lung_scc 7–8 | 12 | 12 | 0 | Örneklemde aynı-kaynak sinyali güçlü kalıyor. |
| lung_scc 9–12 | 12 | 12 | 0 | İncelenen çiftlerin tamamı aynı mikroskobik içeriğe ait görünüyor. |

Bu 12'şer örnek her bandı ayrı ayrı tanımak için seçilmiştir. Bantların gerçek büyüklüğü çok
farklı olduğu için brain için `5/60 = %8,33` sayısı bütün adayların gerçek-akraba oranı gibi
kullanılamaz. Bu oran yalnızca eşit sayıda örnek alınmış görsel inceleme kümesini anlatır.

### Beyin grubundaki beş güçlü aynı-kaynak adayı

1. `brain_glioma_1935.jpg` ↔ `brain_glioma_4817.jpg`, uzaklık 2;
2. `brain_glioma_0971.jpg` ↔ `brain_glioma_4694.jpg`, uzaklık 2;
3. `brain_glioma_0442.jpg` ↔ `brain_glioma_3977.jpg`, uzaklık 2;
4. `brain_glioma_3116.jpg` ↔ `brain_glioma_4520.jpg`, uzaklık 2;
5. `brain_glioma_3222.jpg` ↔ `brain_glioma_3622.jpg`, uzaklık 4.

Bu çiftlerde yalnız genel kafa şekli değil, iç anatomik ayrıntılar ve lezyon/kontrast düzeni de
neredeyse aynıdır. SHA-256'ları farklı olduğundan byte-byte aynı dosya değildirler; yeniden
kodlama veya görüntü dönüşümüyle üretilmiş güçlü akraba adaylarıdır.

Reservoir örneklemesine girmeyen tek beyin uzaklık-0 çifti
`brain_glioma_1366.jpg ↔ brain_glioma_2092.jpg` çiftidir. Görsel olarak çok ilişkilidir ancak
kesin bir parlaklık/döndürme türevi kadar açık değildir. ORB kontrolünde 86 iyi özellik eşleşmesi,
36 geometrik inlier ve 0,4186 inlier oranı vermiştir. Komşu MR kesitleri, aynı çekimin ilişkili
görüntüleri veya yalnızca çok benzer bağımsız kesitler olabilir. Kimlik bilgisi olmadığı için bu
çifti kesin aynı-kaynak diye etiketlemek doğru olmaz.

### Akciğer grubundaki uzaklık-0 fakat SHA-256-farklı çiftler

Standart temas sayfası örneklemesinin dışındaki aşağıdaki sekiz çift de ayrıca görüntülenmiştir:

- `1349 ↔ 2079`, `2286 ↔ 4562`, `0603 ↔ 4742`, `2972 ↔ 3821`;
- `1118 ↔ 3822`, `1860 ↔ 3637`, `1748 ↔ 3982`, `2520 ↔ 3290`.

Sekizinde de aynı mikroskobik alan ve hücre dokusu görülmekte; renk/parlaklık gibi küçük
değişiklikler SHA-256'yı değiştirmiştir. Bu, pHash'in SHA-256'nın bulamadığı görsel akrabaları
bulabildiğine somut örnektir.

## 5. Ek sayısal/görsel doğrulama

Manuel gözlemi tek başına bırakmamak için 120 temas-sayfası çiftinde iki tür destekleyici kontrol
yapılmıştır. Bunlar nihai otomatik sınıflandırıcı veya seçilmiş eşik değildir.

### Piksel korelasyonu, SSIM ve normalize MAE

| Görsel hüküm grubu | n | Korelasyon medyanı (aralık) | SSIM medyanı (aralık) | MAE medyanı |
|---|---:|---:|---:|---:|
| Beyin bağımsız | 55 | 0,6138 (0,4703–0,7958) | 0,6383 (0,4918–0,7338) | 0,0618 |
| Beyin güçlü aynı-kaynak | 5 | 0,8054 (0,6513–0,9684) | 0,6844 (0,6299–0,7140) | 0,0444 |
| Akciğer güçlü aynı-kaynak | 60 | 0,5313 (0,0748–1,0000) | 0,4033 (0,1391–1,0000) | 0,0707 |

Akciğer çiftleri gözle açıkça aynı mikroskobik alan olduğu halde doğrudan korelasyon bazen çok
düşüktür. Bunun nedeni renk/parlaklık, kayma, çevirme veya başka geometri değişikliklerinin aynı
konumdaki pikselleri bozmasıdır. Sonuç: düşük ham-piksel korelasyonu “bağımsızdır” kanıtı değildir.

### ORB özellik eşleşmesi ve RANSAC geometrik tutarlılığı

| Görsel hüküm grubu | İyi ORB eşleşmesi | Geometrik inlier | Inlier oranı |
|---|---:|---:|---:|
| Beyin bağımsız, n=55 | 7–70; medyan 23 | 0–14; medyan 5 | medyan 0,28; en çok 0,57 |
| Beyin güçlü aynı-kaynak, n=5 | 819–1.002; medyan 956 | 788–994; medyan 939 | 0,96–0,99 |
| Akciğer güçlü aynı-kaynak, n=60 | 522–1.500; medyan 909,5 | 498–1.500; medyan 903,5 | 0,95–1,00; medyan 0,99 |

ORB iki görüntüdeki ayırt edici yerel noktaları eşleştirir; RANSAC ise bu eşleşmelerin tek bir
geometrik dönüşümle açıklanıp açıklanmadığını sınar. Aynı-kaynak görülen çiftlerle bağımsız beyin
çiftleri arasında burada çok büyük ayrım vardır ve manuel hükmü kuvvetle destekler. Fakat akciğer
için bağımsız negatif kontrol henüz ölçülmediğinden bu sayılardan evrensel bir ORB eşiği çıkarmak
doğru değildir.

## 6. Hata var mı?

### Çalıştırma veya hesaplama hatası görünmeyen noktalar

- 10.000/10.000 görüntü hash'lendi;
- kombinasyon sayıları matematiksel olarak doğru;
- her sınıfın bütün ikilileri tarandı;
- 120 örnek ve 20 temas sayfası beklenen sayıyla üretildi;
- bilinen 279 SHA-256 tam-kopya çiftinin tamamı pHash 0 verdi;
- rapor herhangi bir görsel okuma veya exact-audit uyarısı içermiyor;
- yalnız çift uzaklıklarının görülmesi hash tasarımının sonucu, arıza değil.

### Sonucu sınırlandıran yöntem sorunları

1. `00-02` bandı uzaklık 0 ve 2'yi birlikte örnekliyor. Reservoir sampling, tek beyin pHash-0
   çiftini ve sekiz SHA-farklı akciğer pHash-0 çiftini temas sayfalarına hiç seçmedi. Gelecek
   sürüm, SHA-256'sı farklı olan bütün pHash-0 çiftlerini zorunlu olarak göstermeli.
2. `03-04` gibi bant adları tek/çift davranışını bilmeyen kullanıcıyı yanıltabilir.
3. Manuel hükümler makine-okunur bir `review_labels.csv` dosyasına kaydedilmedi.
4. Rastgele bağımsız çiftlerden oluşan negatif kontrol grubu yok.
5. Pilot hassasiyete dair fikir verir; kaç gerçek akrabanın 12'nin üstünde kaldığını bilmediğimiz
   için recall/kaçırma oranını ölçmez.
6. Yalnız iki alt sınıf tarandığı için sekiz organ ve farklı görüntü modaliteleri hakkında genel
   eşik kararı verilemez.
7. Hasta, çekim ve patoloji slide kimlikleri yok; hasta düzeyi bağımsızlık kanıtlanamaz.

## 7. Bilimsel sonuç

- **pHash yararlı ama tek başına aile kararı veremez.** Aday üretmek için kullanılmalıdır.
- **Tek global pHash eşiği seçilmemelidir.** Görüntü modalitesi sonucu yaklaşık iki büyüklük
  mertebesi değiştirmiştir.
- **Beyin MR için pHash tek başına risklidir.** Uzaklık 2'de bile 12 örneğin 8'i bağımsız
  görünmüştür; uzaklık 4 örneklerinde 12'nin 11'i false positive'tir.
- **Akciğer histolojisinde pHash umut vericidir.** İncelenen 60 çiftin tamamı aynı-kaynak adayıdır;
  fakat bu, 12 eşiğinin eksiksiz olduğu veya bütün 1.731 çiftin kesin akraba olduğu anlamına gelmez.
- **En güçlü sonraki yöntem hibrittir:** pHash ile geniş aday üretimi, ORB/RANSAC ile geometrik
  doğrulama, elle etiketlenmiş pozitif/negatif kontrol ve modalite bazlı değerlendirme.

Bu nedenle bu pilotun sonucu “eşik 12 seçildi” değildir. Sonuç şudur: araç çalışıyor ve gerçek
yakın kopyaları bulabiliyor; ancak beyin MR ile histoloji için aynı karar kuralını kullanmak yanlış
olur. Yeni bir split manifesti ancak bu hibrit yöntem daha fazla alt sınıf ve negatif kontrollerle
doğrulandıktan sonra dondurulmalıdır.

## 8. Önerilen bir sonraki kontrollü adım

Mehmet'in onayından sonra, model eğitmeden:

1. İncelenen çiftlerin hükümlerini `review_labels.csv` olarak kaydetmek;
2. pHash-0/SHA-farklı çiftleri zorunlu örneğe dahil etmek;
3. her modalite için rastgele bağımsız negatif kontrol çiftleri eklemek;
4. ORB/RANSAC değerlerini tekrar üretilebilir bir denetim çıktısına dönüştürmek;
5. 12'nin üzerindeki uzaklıklardan örnek alıp gerçek akraba kaçırma riskini incelemek;
6. en az birer temsilci alt sınıfla diğer modalitelere genişletmek;
7. ancak bundan sonra aile gruplarını ve sabit 70/15/15 split manifestini oluşturmak.

Bu sırada hiçbir model eğitimi başlatılmamalı ve 97% gibi önceki image-level split sonuçları nihai
performans olarak yorumlanmamalıdır.

### İzlenebilirlik güncellemesi — 2026-10-19

`candidate_samples.csv`, `review_labels.csv` ve 20 temas sayfası aynı `P0001`–`P0120` Pair ID
değerlerini kullanacak şekilde güncellenmiştir. Önceden üretilmiş aday çiftler kullanılarak yalnız
temas sayfaları yeniden çizilmiş; pHash hesaplama ve 24.995.000 çiftlik tarama tekrarlanmamıştır.
Pair ID, görselde verilen kararın inceleme tablosundaki satıra yanlış eşlenmesini önler.
