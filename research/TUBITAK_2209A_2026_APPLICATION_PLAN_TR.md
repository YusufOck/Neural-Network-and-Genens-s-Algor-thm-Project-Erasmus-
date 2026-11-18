# TÜBİTAK 2209-A 2026 Başvuru Planı — NNGA Multi-Cancer Projesi

Son doğrulama: 2026-11-18

## 1. Çağrı durumu

- Güncel TÜBİTAK program sayfasına göre başvuru ve akademik danışman onayı son tarihi **23 Kasım 2026, saat 17.30**.
- Kuruluş yetkilisi e-imza dönemi **26 Kasım–3 Aralık 2026**.
- Başvuru TYBS üzerinden proje yürütücüsü öğrenci tarafından çevrim içi yapılır.
- Araştırma önerisi TYBS alanlarına girilen bilgilerden otomatik oluşturulur; ayrı bir PDF öneri yükleme adımı bulunmaz.

## 2. Ön uygunluk kontrolü

Başvuruya geçmeden önce şu dört madde teyit edilmelidir:

1. Yürütücü, Türkiye'de lisans veya ön lisans derecesi veren bir programa aktif kayıtlı mı?
2. Yürütücünün ARBİS hesabı güncel mi?
3. Akademik danışmanın güncel YÖK kaydı var mı ve TYBS/YÖKSİS/ARBİS bilgilerinde görünüyor mu?
4. Yürütücü aynı dönemde başka 2209-A/2209-B projesinde yürütücü veya ortak değil mi?

Erasmus öğrencisi olmak tek başına yeterli veya engel değildir. Esas kontrol, Türkiye'deki lisans/ön lisans kaydının aktif olup olmadığıdır. Yalnızca Romanya'daki üniversite kaydı varsa uygunluk ayrıca TÜBİTAK BİDEB'den teyit edilmelidir.

## 3. Önerilen proje çerçevesi

**Geçici başlık:**

> Çok Modaliteli Kanser Görüntülerinde Bilinen Kopya Aileleri Ayrılmış 26 Alt Sınıflı Görüntü Sınıflandırması ve Sınırlı Hiperparametre Optimizasyonu

Bu başlık klinik tanı iddiası taşımaz. Veri setindeki etiketleri, organ/modality kaynaklarını ve 26 alt sınıfı sınıflandırma problemi olarak tanımlar.

### Ön çalışma kanıtı

- 130.000 görüntü, 26 alt sınıf ve sekiz üst veri grubu envanteri çıkarıldı.
- Bilinen exact/review bağlantıları aynı ailede tutulacak şekilde sabit manifest oluşturuldu: 91.000 train, 19.500 validation, 19.500 test.
- Aynı test yollarında üç kontrollü model koşusu ve Grad-CAM denetimi tamamlandı.
- Bounded GA ile eşit bütçeli random-search karşılaştırması proxy validation üzerinde tamamlandı; bu sonuç final test başarısı veya genel GA üstünlüğü olarak yazılmayacak.

### Araştırma sorusu

> Bilinen kopya ve incelenmiş yakın-aile bağlantıları splitler arasında tutulmadığında, 26 alt sınıflı görüntü sınıflandırma modellerinin sonuçları nasıl değişir ve sınırlı/eşit bütçeli hiperparametre araması sabit bir ResNet18 kapsamını iyileştirir mi?

### Yöntem özeti

1. Dondurulmuş 26-leaf manifesti kullanmak.
2. ResNet18, SmallCNN ve last-block fine-tuning sonuçlarını aynı test yollarında karşılaştırmak.
3. Accuracy yanında macro-F1, balanced accuracy, per-class recall ve confusion matrix raporlamak.
4. GA ve random search'ü yalnız validation ile seçmek; test setini seçime kapalı tutmak.
5. Grad-CAM ile sınırlı, post-hoc görsel açıklanabilirlik denetimi yapmak.
6. Sonuçları klinik tanı olarak değil, veri setine özgü teknik sınıflandırma sonucu olarak yorumlamak.

## 4. TYBS'de hazırlanacak içerik

- Özet: 25–450 kelime; bilimsel nitelik, yöntem, proje yönetimi ve yaygın etki içermeli.
- Bilimsel nitelik ve önem: en az 350 karakter.
- Amaç ve hedefler: en az 350 karakter; ölçülebilir ve gerçekçi.
- Yöntem: en az 350 karakter; tasarım, değişkenler, metrikler, çalışma takvimi ve ön çalışma.
- Çalışma takvimi: iş paketleri, süre, başarı ölçütü; başarı ölçütleri toplamı %100.
- Risk yönetimi: veri sızıntısı, sınıf karışması, GPU/Colab kesintisi ve B planları.
- Araştırma olanakları: GPU/Colab, yerel bilgisayar, GitHub ve veri arşivi.
- Yaygın etki: tekrarlanabilir kod, rapor, görsel açıklama ve bilimsel sunum çıktıları.
- Diğer konular/ekler: yalnız değerlendirmeye katkı sağlayan tablo, akış veya görsel.
- Kaynakça: metin içi atıflar ve DOI'ler; Türkçe öneri metni.

## 5. Bütçe notu

Güncel program sayfası azami desteği 12.000 TL olarak veriyor. Makine/teçhizat, sarf, seyahat ve hizmet alımı kapsam dahilinde; konferans/yayın/patent, konaklama ve gündelik kapsam dışıdır. Bu proje için bütçe ancak gerçekten gerekli ve fatura ile belgelendirilebilir kalemlerden oluşmalıdır.

## 6. Eksik kişisel bilgiler

Başvuru metnini kişiselleştirmek için şu alanlar yürütücüden alınmalıdır:

- Türkiye'deki üniversite, fakülte/bölüm ve aktif öğrenci durumu
- Proje yürütücüsü adı ve e-posta
- Akademik danışman adı, bölüm ve e-posta
- Varsa en fazla dört proje ortağı
- Talep edilecek gerçek bütçe kalemleri
- Proje için üniversitenin etik kurul/muafiyet prosedürü

Tıbbi görüntüler kamuya açık olsa bile etik kurul gerekip gerekmediği varsayılmayacak; üniversitenin etik biriminden muafiyet/uygunluk görüşü alınacaktır.

## 7. Başvuru sırası

1. Uygunluk ve danışman teyidi.
2. TYBS/ARBİS/YÖKSİS bilgilerini kontrol etme.
3. Bu taslağı kişisel bilgilerle doldurma.
4. TYBS alanlarına Türkçe metni girme ve önizleme yapma.
5. 23 Kasım 17.30'dan önce yürütücü son onayı ve danışman onayını tamamlama.
6. 26 Kasım–3 Aralık arasında kuruluş yetkilisi e-imza durumunu takip etme.

Bu plan başvuru hazırlığıdır; başvuru yapıldı veya destek alındı anlamına gelmez.
