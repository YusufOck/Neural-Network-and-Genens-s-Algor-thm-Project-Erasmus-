# NNGA proje durum kaydı

Son güncelleme: 2026-10-25

Bu dosya, küçük bir pilotun yanlışlıkla bütün veri setinde tamamlanmış bir denetim gibi
anlatılmasını önlemek için projenin tek durum kontrol listesidir. Bir adım ancak aşağıdaki
tanımlara göre işaretlenir.

## Durumların anlamı

- **Tüm kapsamda doğrulandı:** Amaçlanan veri kapsamının tamamı işlendi, çıktı kaydedildi ve
  gerekli kontroller yapıldı.
- **Pilot tamamlandı:** Yöntem yalnızca açıkça belirtilen küçük/temsilci kapsamda çalıştırıldı.
  Bütün veri seti için sonuç sayılmaz.
- **Devam ediyor:** İşin bazı parçaları tamamlandı fakat kabul koşulları henüz karşılanmadı.
- **Başlamadı:** Kod taslağı veya düşünce bulunabilir; doğrulanmış uygulama henüz yoktur.
- **Öğrenme bekliyor:** Teknik çıktı hazır olsa bile Mehmet ile açıklama/geri anlatım kontrolü
  tamamlanmamıştır.

## 1. Kapsam ve etiket anlamı

| İş | Durum | Kapsam ve eksik |
|---|---|---|
| Profesör konuşmasının çözümlenmesi | Tüm kapsamda doğrulandı | Transcript incelendi; sekiz sınıflı image classification istendiği kaydedildi. |
| 8 üst grup / 26 alt sınıf envanteri | Tüm kapsamda doğrulandı | `config/label_taxonomy.json` ve ana veri denetim raporu hazır. |
| 26 etiketin Mehmet tarafından savunulabilecek düzeyde öğrenilmesi | Öğrenme bekliyor | Teknik tablo hazır; bütün etiketlerin anlamı birlikte tekrar edilmedi. |
| Nihai hedefin profesörle doğrulanması | Başlamadı / bilinçli ertelendi | Mehmet şimdilik mesaj göndermeme kararı verdi. Final deneyden önce yeniden değerlendirilecek. |

## 2. Bütün veri setine uygulanmış temel denetimler

Bu bölümdeki sonuçlar yalnızca iki sınıfa değil, 130.000 kabul edilen görüntünün tamamına aittir.

| İş | Durum | Kanıt |
|---|---|---|
| Dosya sayımı ve 8/26 klasör yapısı | Tüm kapsamda doğrulandı | 26 alt sınıf × 5.000 = 130.000 görüntü. |
| Görüntü açılabilirlik/bozulma kontrolü | Tüm kapsamda doğrulandı | 130.000/130.000 kabul edilen görüntü açıldı; bozuk görüntü yok. |
| Boyut, kanal ve gerçek dosya formatı | Tüm kapsamda doğrulandı | RGB; 129.997 adet 512×512, üç adet 640×640; 5.000 PNG içerikli `.jpg` belirlendi. |
| SHA-256 exact-duplicate denetimi | Tüm kapsamda doğrulandı | 1.318 grup, 2.673 grup üyesi, 1.355 fazladan birebir kopya; sınıflar arası exact grup yok. |
| Kaynak veri/provenance araştırması | Devam ediyor ama ana envanter hazır | Kaynak veri sayıları ve bilinen augmentation bilgisi kaydedildi; hasta/slide kimlikleri yerel dosyalardan geri getirilemiyor. |

## 3. Yakın kopya ve augmentation-family denetimi

Şu anda bulunduğumuz aşama burasıdır.

| Alt adım | Durum | Tam kapsam / eksik |
|---|---|---|
| pHash pilot kodu | Pilot altyapısı tamamlandı | Aday üretimi, uzaklık histogramı, örnekleme ve temas sayfaları çalışıyor. |
| İki-alt-sınıf pHash taraması | Pilot tamamlandı | Yalnızca `brain_glioma` ve `lung_scc`: 10.000 görüntü, 24.995.000 sınıf-içi çift. |
| Pair ID ve inceleme tablosu | Pilot altyapısı tamamlandı | `P0001`–`P0120`; CSV/JSON/temas sayfası eşleşme hatası 0. |
| SHA-256 pozitif kontrolü | Pilot kapsamında tamamlandı | Pilot içindeki 279 exact `lung_scc` çiftinin tamamı pHash 0. |
| İlk görsel yorum | Pilot kapsamında ön inceleme tamamlandı | Codex destekli ön inceleme raporlandı; bu, resmi ortak etiket tablosunun tamamlandığı anlamına gelmez. |
| Eski 120-pair pilotun izlenebilir etiketlenmesi | Devam ediyor / pilot kaydı | 8 exact satır hazır, 112 satır bekliyor; bu pHash-yanlı iki-sınıf örneklemi nihai kalibrasyon kümesi değildir. |
| Bütün 26 sınıflı resmi kalibrasyon kümesi | Development tamamlandı / holdout korumalı | 1.342 çift: 1.064 kesin pozitif ve 278 inceleme adayı; development etiket/skorları hazır, holdout yöntem dondurulana kadar kapalı. |
| Kalibrasyon insan-inceleme paketi | Development incelemesi tamamlandı / holdout kapalı | 278 Pair ID ve 64 temas sayfası hazır. 139/139 development kararı AI-destekli olarak kaydedildi; 139/139 holdout boş tutuldu. |
| Bağımsız negatif kontroller | Development doğrulandı / holdout bekliyor | Development içinde 123 bağımsız, 15 aynı-kaynak ve 1 belirsiz karar kaydedildi. Bu oran doğal duplicate prevalence değildir. |
| Tekrar üretilebilir ORB/RANSAC doğrulaması | Development pilotu tamamlandı / rule v1 donduruldu | 671/671 development çifti için kalıcı kod, CSV ve JSON çıktı üretildi; bu kanıtla birleşik rule v1 donduruldu ve holdout açılmadı. |
| Birlesik duplicate karar kurali v1 | Development kurali donduruldu / holdout kapali | Uc cikisli `same_source` / `independent` / `uncertain` kurali JSON'a yazildi ve SHA-256 degeri kaydedildi. Development'ta 547/547 ayni-kaynak, 123/123 bagimsiz karar korundu; C000620 uncertain kaldi. Bu holdout basarisi degildir. |
| Uzaklık 12 üzerindeki recall/kaçırma kontrolü | Kontrollü sentetik kapsamda tamamlandı | 1.040 kesin sentetik pozitifin 552'si (%53,08) pHash 12 üzerindeydi; doğal akrabalar ve diğer aday kanalları henüz ölçülmedi. |
| Kan yayması mikroskopisi doğrulaması | Development pilotu tamamlandı / holdout bekliyor | Kontrollü pozitifler ve development review çiftleri skorlandı; bağımsız ölçüm yapılmadı. |
| Pap smear mikroskopisi doğrulaması | Development pilotu tamamlandı / holdout bekliyor | ORB/phase/ECC kanıtları çıkarıldı; beş bağımsız çiftte ECC yakınsamadı. |
| CT doğrulaması | Development pilotu tamamlandı / holdout bekliyor | `kidney_normal` ve `kidney_tumor` development çiftleri skorlandı; `C000620` belirsiz kaldı. |
| Diğer MRI alt sınıfları | Development pilotu tamamlandı / holdout bekliyor | Üç brain yaprağı development kalibrasyonunda temsil edildi; bağımsız ölçüm yapılmadı. |
| Diğer H&E histopatoloji alt sınıfları | Development pilotu tamamlandı / holdout bekliyor | Bütün H&E kaynak grupları development kalibrasyonunda temsil edildi; bağımsız ölçüm yapılmadı. |
| Yöntemin 26 alt sınıfın tamamına uygulanması | Başlamadı | Pilot ve modalite doğrulaması bitmeden çalıştırılmayacak. |
| Sınıflar arası perceptual-duplicate kontrolü | Başlamadı | Pilot yalnız aynı alt sınıf içindeki çiftleri karşılaştırdı. Uyumlu modalite/kaynak grupları arasında değiştirilmiş çapraz-etiket kopyaları ayrıca aranmalı. |
| 130.000 görüntü için augmentation-family grupları | Başlamadı | Henüz hiçbir nihai aile listesi yok. |

## 4. Split durumu

| İş | Durum | Neden |
|---|---|---|
| Sabit naive 70/15/15 manifest | Başlamadı | Önce hedef ve veri protokolü kesinleştirilecek. Eski pilotların rastgele split'i final manifest değildir. |
| Exact-duplicate-aware manifest | Başlamadı | SHA-256 grupları hazır fakat kalıcı manifest üretilmedi. |
| Near-duplicate-family-aware manifest | Başlamadı | Yakın-kopya aileleri henüz oluşturulmadı. |
| Split sızıntı doğrulaması | Başlamadı | Manifestler oluştuktan sonra train/validation/test arasında aile kesişimi 0 olmalı. |

## 5. Model ve sonraki aşamalar

| İş | Durum | Açıklama |
|---|---|---|
| Trainer altyapısı | Teknik temel var; final hazır değil | 8/26 hedef ve dört mimari destekleniyor; split ve kayıt eksikleri var. |
| SmallCNN smoke test | Smoke test tamamlandı | Yalnızca sistemin uçtan uca çalıştığını gösterir; performans sonucu değildir. |
| ResNet18 8-grup pilotu | Pilot tamamlandı | Rastgele image-level split nedeniyle final sonuç değildir. |
| DenseNet121 26-sınıf küçük reproduction | Pilot/reproduction tamamlandı | Küçük örnek ve rastgele split; final sonuç değildir. |
| Adil model karşılaştırması | Başlamadı | Sabit leakage-aware split bekleniyor. |
| Genetik algoritma | Başlamadı | Stabil baseline sonrasına bırakıldı. |
| Grad-CAM | Başlamadı | Final/seçilmiş modellerden sonra. |
| Rapor ve sunum | Başlamadı; kanıtlar birikiyor | Denetim raporları daha sonra ana rapora girdi olacak. |

## 6. Değiştirilemez ilerleme sırası

1. `research/NEAR_DUPLICATE_METHOD_PLAN_TR.md` yöntem ve kalibrasyon tasarımını Mehmet ile onayla. **Tamamlandı.**
2. Bütün 26 alt sınıftan exact/sentetik pozitif, rastgele/hard negative, >12 ve çapraz-sınıf
   örnekleri içeren `calibration-set v1` oluştur. Mevcut 120 çifti ayrı pilot kaydı olarak koru.
   **Taslak üretildi; 139 development adayı incelendi, 139 holdout adayı yöntem dondurulana kadar
   kapalı tutuluyor.**
3. Yeni şemayla resmi çift etiketlemesini yap; eski pilotun 112 bekleyen satırını da pilot kaydı
   olarak kapat, fakat nihai kalibrasyon sonucu sayma. **Yeni kalibrasyonda 139/139 development
   kararı AI-destekli olarak girildi; eski pilot kaydı hâlâ açık, holdout yöntem dondurulana kadar
   kapalı.**
4. pHash aday üretimi, ORB/RANSAC, ECC/phase correlation ve hizalanmış SSIM/NMI yöntemlerini
   tekrar üretilebilir kod ve CSV çıktısına dönüştür. **Development özellik çıkarımı tamamlandı;
   birleşik karar kuralı v1, holdout açılmadan önce JSON ve SHA-256 ile donduruldu.**
5. Kalibrasyon ile eşikleri geliştir; farklı kaynak görüntülerden oluşan hold-out çiftlerinde beş
   ana modaliteyi doğrula: kan yayması, MRI, H&E histopatoloji, Pap smear ve CT.
6. Aynı denetim protokolünü 26 alt sınıfın tamamına uygula. Modaliteye göre farklı karar kuralı
   gerekebilir; hiçbir alt sınıf kapsam dışı bırakılamaz.
7. Uyumlu görüntü modaliteleri/kaynakları arasında sınıflar arası perceptual-duplicate adaylarını
   denetle. Bunu bütün 130.000 dosyanın kör `N²` karşılaştırması yerine doğrulanmış aday-arama
   yöntemiyle yap.
8. Exact ve doğrulanmış yakın-kopyaları aile gruplarına dönüştür.
9. Aileleri bölmeden sabit 70/15/15 manifestleri oluştur ve kesişim testini yap.
10. Ancak bundan sonra adil model eğitimi ve karşılaştırmasına geç.

Bir pilotta bulunan eşik veya kural, yukarıdaki 2–5. adımlarda doğrulanmadan bütün veri setine
uygulanamaz. Buna karşılık, nihai split hazırlanırken 26 alt sınıftan hiçbiri yalnızca pilotta yer
almadığı gerekçesiyle denetimsiz bırakılamaz.
