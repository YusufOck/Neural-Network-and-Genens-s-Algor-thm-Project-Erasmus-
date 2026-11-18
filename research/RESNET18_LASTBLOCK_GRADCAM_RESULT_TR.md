# ResNet18 Last-Block Grad-CAM Audit Sonucu

Tarih: 2026-11-18

Durum: Gerçek Colab/T4 çalışması tamamlandı; paket bütünlüğü ve 37 panelin görsel denetimi tamamlandı.

Deney türü: `post_hoc_explainability_audit`

## 1. Sonuç cümlesi

Fine-tuned ResNet18'in dondurulmuş 37 test örneğindeki Grad-CAM haritaları başarıyla üretildi ve
model tahminleri 37/37 örnekte yeniden üretildi. Haritalar çoğu doğru kontrolde görüntü içindeki
doku, hücre veya organ bölgelerine yöneliyor; ancak uzamsal çözünürlükleri kaba ve bazı örneklerde
kenar, boş arka plan veya kaynak-görüntü şerit artefaktları da etkili. Yanlış örneklerde tahmin
edilen ve gerçek sınıf haritaları sıklıkla aynı veya yakın doku bölgelerini vurguluyor. Bu nedenle
sonuç, modelin tamamen anlamsız arka plana baktığını göstermiyor; fakat kararların tıbben doğru,
nedensel veya dış veride güvenilir olduğunu da kanıtlamıyor.

## 2. Dondurulmuş sözleşme

- Model checkpoint SHA-256:
  `538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d`
- Split manifest SHA-256:
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`
- Test tahminleri SHA-256:
  `237f3283b88beeaf0a93f182adab23ad478810b0944245e3023e6cfe885c57bd`
- Örnek manifesti SHA-256:
  `3f797979b4d75b092dfcb4e2348ea1705ed34d35da884ac999c68abef8c87add`
- Hedef katman: `layer4.1.conv2`
- Cihaz: Tesla T4 / CUDA
- Görüntü seçimi ısı haritaları görülmeden önce donduruldu.

Seçim 26 leaf'in her birinden bir doğru kontrol, yedi `confidence >= 0,99` hata, iki cross-parent
hata ve iki yönlü oral normal/SCC medyan hata örneğinden oluşur. Yol tekrarları kaldırıldıktan sonra
toplam 37 benzersiz test görüntüsü vardır.

## 3. Artifact doğrulaması

- Review ZIP SHA-256:
  `4f2bc74fe0e57986749371dd93f0d86ed929a4afecbf3a34f7bd93ade0b65b92`
- ZIP boyutu: 34.061.019 byte
- İç hash manifesti: 48 kayıt
- Yeniden doğrulanan kayıt: 48/48
- Hash veya boyut uyuşmazlığı: 0
- Manifest dışında beklenmeyen dosya: 0
- Sonuç satırı: 37
- Bireysel panel: 37
- Temas sayfası: 7
- Tahmin yeniden üretimi: 37/37
- En büyük confidence mutlak farkı: `3,4570693969726562e-06`

Bu fark önceden tanımlanan `5e-5` toleransının oldukça altındadır. Dolayısıyla haritaların
üretildiği checkpoint çalışması, dondurulmuş test tahminleriyle uyumludur.

## 4. Panel nasıl okunur?

Doğru kontrol panelinde solda model girdisi, ortada hedef sınıfın Grad-CAM haritası ve sağda
görüntü üzerindeki overlay vardır. Yanlış panelde solda girdi, ortada modelin yanlış tahmin ettiği
sınıfın haritası ve sağda gerçek sınıf skorunun haritası bulunur. Kırmızı/sarı daha yüksek pozitif
etkiyi, mavi daha düşük etkiyi gösterir. Renkler sınıflar arası olasılık değildir ve kırmızı bölge
"kanser burada" anlamına gelmez.

## 5. Yedi temas sayfasının ve 37 panelin görsel denetimi

### 5.1 Doğru kontroller

- ALL kan-yayması örneklerinde vurgu çoğunlukla hücre içeren orta/alt alanlarda toplanır. Bununla
  birlikte bazı haritalar geniş bir alanı kaplar ve girdi kenarındaki görünür şeritlere yaklaşır.
- Brain MRI kontrollerinde vurgu genel olarak kafatası içindeki beyin bölgesindedir. Meningioma ve
  bazı tümör örneklerinde parlak kitleyle örtüşme vardır; fakat harita kitle sınırı değildir ve
  çevrede geniş bir beyin alanını da kapsar.
- Breast, cervix, colon ve lung mikroskopi kontrollerinde aktivasyon çoğunlukla hücre/doku içeren
  alanlara düşer. Bazı örneklerde en sıcak bölge doku sınırına veya görüntü köşesine yakındır.
- Kidney CT kontrollerinde aktivasyon geniş abdominal bölgeyi kaplar. Tümör örneğinde görünür
  kitleyle yaklaşık örtüşme vardır; bu yalnız kaba model etkisidir.
- Lymphoma ve oral histoloji kontrollerinde ana aktivasyon dokunun üzerindedir; fakat bazı
  görüntülerde kaynak veride bulunan kenar-uzatma/şerit artefaktlarına yakın güçlü aktivasyon da
  görülür. Bu, modelin yalnız hücresel morfoloji kullandığının iddia edilemeyeceğini gösterir.

### 5.2 On bir yanlış tahmin

- `ERROR_01` ve `ERROR_02` gerçek `brain_glioma`, tahmin `brain_menin` örnekleridir. Tahmin haritası
  lezyon/merkez çevresinde yoğunlaşırken gerçek-sınıf haritası başka ve daha geniş beyin bölgelerine
  kayar. `ERROR_01` gerçek-sınıf haritasında görüntünün alt dış bölgesine taşan aktivasyon da vardır.
- `ERROR_03`, `ERROR_05` ve `ERROR_06` breast benign/malignant karışmalarıdır. İki hedef haritası
  farklı doku alt bölgelerine ağırlık verse de her ikisi de aynı histoloji alanının geniş parçalarını
  kullanır. Bu örneklerin yüksek confidence ile yanlış olması, sıcak bölgenin doğru karar garantisi
  olmadığını açıkça gösterir.
- `ERROR_07` gerçek `cervix_mep`, tahmin `cervix_koc` örneğinde tahmin ve gerçek hedefler farklı hücre
  kümelerine yönelir. Haritalar yine hücre düzeyinde keskin değil, geniş ve kaba bölgelerdir.
- `ERROR_09`, `ERROR_10` ve `ERROR_11` oral normal/SCC karışmalarıdır. Tahmin ve gerçek hedef
  haritaları büyük ölçüde aynı doku bölgelerini vurgular; fark çoğunlukla sıcaklığın iki odak
  arasındaki dağılımıdır. Bu, modelin bu çift için benzer görsel kanıt kullandığıyla uyumludur.
- `ERROR_04` gerçek breast benign, tahmin oral normal olan cross-parent hatadır. Tahmin haritası dar
  doku şeridine, gerçek-sınıf haritası ise önemli ölçüde boş arka plan/sınır bölgesine yayılır. Bu,
  görüntü kompozisyonu veya kaynak artefaktı kısayolu açısından en güçlü uyarı örneğidir.
- `ERROR_08` gerçek lung SCC, tahmin breast malignant olan cross-parent hatadır. İki hedef haritası
  aynı histoloji dokusunun çok benzer alt bölgelerini vurgular. Bu durum organ etiketinden bağımsız
  ortak histoloji görünümünün modele kısayol sağlayabileceği hipoteziyle uyumludur; tek başına bunu
  kanıtlamaz.

## 6. Bilimsel yorum

Bu audit'ten savunulabilir üç sonuç çıkar:

1. Model birçok örnekte tamamen rastgele boşluğa değil, görüntüdeki doku/organ/hücre alanlarına
   duyarlıdır.
2. Modelin etkisi bazı örneklerde kenar, boşluk ve kaynak-görüntü artefaktlarına da uzanır; bu nedenle
   yüksek test başarısı yalnız hastalık morfolojisinin öğrenildiği anlamına gelmez.
3. Yanlış tahminlerde iki sınıfın haritaları sık sık birbirine benzediği için Grad-CAM sınıf
   ayrımının tıbben doğru yapıldığını doğrulayamaz; yalnız hangi kaba bölgelerin skoru etkilediğini
   gösterir.

Bu sonuç model performans metriği değildir. Önceden raporlanan test accuracy veya macro-F1 değerini
artırmaz, azaltmaz ve model seçimini değiştirmez.

## 7. Sınırlılıklar

- Son convolution katmanının uzamsal çözünürlüğü düşük olduğu için haritalar kabadır.
- Grad-CAM pozitif sınıf etkisini yaklaşık gösterir; nedensellik, segmentasyon veya nesne kutusu
  vermez.
- Örnekler test setinden post-hoc incelenmiştir; yeni bir dış doğrulama değildir.
- 37 örnek bütün 19.500 test görüntüsündeki açıklama davranışını kanıtlamaz.
- Görsel yorum AI-destekli ve uzman-olmayan bir mühendislik incelemesidir; patolog/radyolog
  doğrulaması değildir.
- Veri setinde görünen kenar-uzatma/şerit artefaktları bazı haritaların yorumunu sınırlar.
- Haritalar etiketin doğru veya yanlış olduğunu belirleyemez.

## 8. Çalışma uyarıları

Colab koşusunda iki uyarı üretildi: gradient taşıyan bir tensörün `float` dönüşümü ve Pillow
`mode="L"` deprecation uyarısı. Çalışma 37/37 tahmini yeniden üretti, bütün dosyaları yazdı ve hash
doğrulaması geçti; bu iki uyarı mevcut bilimsel sonucu bozmadı. Gelecek kod bakımında tensör
`detach().item()` ile okunacak ve deprecated Pillow argümanı kaldırılacaktır. Bu bakım, bu koşunun
hash/provenance zincirini değiştirmeden ayrı commit olarak yapılmalıdır.

## 9. Saklanan kanıtlar

Repo içinde özet JSON, sonuç CSV'si, dondurulmuş config/manifest, iç hash manifesti ve yedi temas
sayfası şu dizinde saklanır:

`research/model_results/resnet18_lastblock_finetune_v1/gradcam_v1/`

37 yüksek çözünürlüklü bireysel paneli içeren tam ZIP Drive'da ve yerel Downloads klasöründe
korunur. Büyük bireysel panel dosyaları repo boyutunu gereksiz büyütmemek için Git'e eklenmemiştir.

## 10. Projedeki yeri

Grad-CAM açıklanabilirlik audit'i tamamlanmıştır. Bir sonraki yeni deney otomatik olarak başlamaz.
Sıradaki bilimsel karar, genetik algoritmanın hangi sınırlı hiperparametreleri hangi eşit hesaplama
bütçesiyle optimize edeceğinin ve aynı bütçeli random-search karşılaştırmasının öğrenilip
dondurulmasıdır.
