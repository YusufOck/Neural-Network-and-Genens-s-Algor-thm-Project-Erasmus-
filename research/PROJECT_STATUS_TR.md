# NNGA proje durum kaydı

Son güncelleme: 2026-10-29

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
| Bütün 26 sınıflı resmi kalibrasyon kümesi | Development ve holdout değerlendirmesi tamamlandı | 1.342 çiftin 671 development ve 671 holdout bölümü işlendi; bu yöntem kalibrasyon/evaluation kümesidir, 130.000 görüntülük tarama değildir. |
| Kalibrasyon insan-inceleme paketi | Tüm 278 aday incelendi | Development 139/139 ve score-blind holdout 139/139 AI-destekli karar kaynağına sahiptir. Holdout kararı: 15 aynı-kaynak, 124 bağımsız. |
| Bağımsız negatif kontroller | Holdout ile doğrulandı | Holdout'taki 124 bağımsız referansın 120'si kesin `independent`, dördü güvenli `uncertain`; yanlış `same_source` 0. |
| Tekrar üretilebilir ORB/RANSAC doğrulaması | Holdout ile doğrulandı | Aynı scorer hash'iyle 671/671 holdout çifti skorlandı; kesinti sonrası 671 karar kayıtlı skorlardan yeniden üretildi. |
| Birleşik duplicate karar kuralı v1 | Dondurulmuş holdout değerlendirmesi tamamlandı | 547/547 aynı-kaynak bulundu; 120 bağımsız kesin ayrıldı; 4 MRI/CT çifti `uncertain`; kesin kararlarda hata 0. Bu kanser sınıflandırma başarısı değildir. |
| Uzaklık 12 üzerindeki recall/kaçırma kontrolü | Kontrollü sentetik kapsamda tamamlandı | 1.040 kesin sentetik pozitifin 552'si (%53,08) pHash 12 üzerindeydi; doğal akrabalar ve diğer aday kanalları henüz ölçülmedi. |
| Kan yayması mikroskopisi doğrulaması | Holdout tamamlandı | 102 çift; `uncertain` 0, kesin kararlarda hata 0. |
| Pap smear mikroskopisi doğrulaması | Holdout tamamlandı | 130 çift; `uncertain` 0, kesin kararlarda hata 0. |
| CT doğrulaması | Holdout tamamlandı | 51 çift; iki bağımsız çift güvenli `uncertain`, kesin kararlarda hata 0. |
| Diğer MRI alt sınıfları | Holdout tamamlandı | 75 çift; iki bağımsız çift güvenli `uncertain`, kesin kararlarda hata 0. |
| Diğer H&E histopatoloji alt sınıfları | Holdout tamamlandı | 313 çift; `uncertain` 0, kesin kararlarda hata 0. |
| Ölçeklenebilir aday üretimi v1 | Pilot tamamlandı / tam taramaya kabul edilmedi | 1.748 kalibrasyon görüntüsü: pHash ≤32 %100 recall fakat çiftlerin %94,5'i; ResNet18 top-50 %99,54 recall ve 5 kaçırma. |
| Ölçeklenebilir aday üretimi v2 | Kalibrasyon pilot kapısı geçti | 1.748 görüntüde robust top-10: 1.094/1.094 aynı-kaynak, 30/30 doğal çift; 18.012/224.926 (%8,01) aday. Tam veri sonucu değildir. |
| Daha büyük retrieval/capacity pilotu | Pilot tamamlandı / robust top-10 kapısı geçmedi | 8.248 görüntüde top-10 1.092/1.094; iki `brightness_1_20` çifti rank 24/27. 122.211 aday = 14,82 çift/görüntü. |
| Post-hoc pHash≤8 safety kontrolü | Exploratory tamamlandı / bağımsız doğrulama değil | `top-10 ∪ pHash≤8` iki scaled-pilot kaçağını 817 ek adayla geri aldı ve v2.1'i motive etti; sonraki frozen v2.1 holdout sonucu aşağıdadır. |
| Candidate Generator v2.1 kaynak-ayrık retrieval holdout | Frozen holdout tamamlandı / kabul kapısı geçmedi | Önceki yollar ve exact gruplarla kesişmeyen 7.644 görüntüde v2.1 birleşimi 1.039/1.040 kontrollü pozitif buldu; tek `combined_stress` kaçağı nedeniyle tam-recall kapısı geçmedi. Yük 14,991 çift/görüntü. |
| Candidate Generator v2.2 ikinci kaynak-ayrık retrieval holdout | Frozen holdout tamamlandı / kabul kapısı geçmedi | Önceki scaled pilot ve v2.1 yollarıyla kesişmeyen 7.644 görüntüde `top-15 ∪ pHash≤8`, 1.039/1.040 kontrollü pozitifi buldu. Tek `brightness_1_20` kaçağı nedeniyle tam-recall kapısı geçmedi; yük 21,765 çift/görüntü. |
| Candidate Generator v2.3 hipotezi | Post-hoc plan / dondurulmadı | V2.2 sonucu görüldükten sonraki tanısal kontrolde `top-15 ∪ pHash≤12`, 1.040/1.040 ve 22,501 çift/görüntü sağladı. Bu aynı holdout üzerindeki geliştirme kanıtıdır; bağımsız doğrulama veya tam-tarama izni değildir. |
| Natural near-duplicate audit v1 | Kör görsel inceleme tamamlandı / scorer karşılaştırması bekliyor | 58 sayfadaki 345/345 çift skorlar açılmadan incelendi: 21 aynı-kaynak adayı, 322 bağımsız-benzer ve 2 `uncertain`. Ayrı karar CSV'si dondurulmaya hazır; private stratum/scorlar henüz açılmadı. |
| Yöntemin 26 alt sınıfın tamamına uygulanması | Başlamadı | Kalibrasyon bütün 26 yaprağı temsil ediyor; fakat 130.000 görüntülük ölçekli aday taraması henüz yapılmadı. |
| Sınıflar arası perceptual-duplicate kontrolü | Başlamadı | Kalibrasyondaki 35 seçilmiş cross-leaf aday tam tarama değildir; uyumlu kaynak grupları ölçekli olarak ayrıca aranmalı. |
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
   **Tamamlandı; 139 development ve score-blind 139 holdout adayı incelendi.**
3. Yeni şemayla resmi çift etiketlemesini yap; eski pilotun 112 bekleyen satırını da pilot kaydı
   olarak kapat, fakat nihai kalibrasyon sonucu sayma. **Yeni kalibrasyonda 139/139 development ve
   139/139 holdout kararı AI-destekli olarak girildi; eski pilot kaydı ayrı açık borç olarak duruyor.**
4. pHash aday üretimi, ORB/RANSAC, ECC/phase correlation ve hizalanmış SSIM/NMI yöntemlerini
   tekrar üretilebilir kod ve CSV çıktısına dönüştür. **Development özellik çıkarımı tamamlandı;
   birleşik karar kuralı v1, holdout açılmadan önce JSON ve SHA-256 ile donduruldu.**
5. Kalibrasyon ile eşikleri geliştir; farklı kaynak görüntülerden oluşan hold-out çiftlerinde beş
   ana modaliteyi doğrula: kan yayması, MRI, H&E histopatoloji, Pap smear ve CT.
   **Tamamlandı; 671/671 holdout çifti tek seferlik frozen-rule değerlendirmesinden geçti.**
6. Aynı denetim protokolünü 26 alt sınıfın tamamına uygula. Modaliteye göre farklı karar kuralı
   gerekebilir; hiçbir alt sınıf kapsam dışı bırakılamaz. **Candidate generator v2 kalibrasyon
   kapısını geçti; 8.248 görüntülük scaled pilotta robust top-10 iki pozitif kaçırdı. Frozen v2.1
   (`top-10 ∪ pHash≤8`) ilk kaynak-ayrık holdout'ta 1.039/1.040 buldu ve tek `combined_stress`
   çifti kaçırdı. Sonucu görmeden dondurulan v2.2 (`top-15 ∪ pHash≤8`) ikinci yeni 7.644
   görüntülük holdout'ta yine 1.039/1.040 buldu; bu kez `brightness_1_20` çifti kaçtı. Post-hoc
   `top-15 ∪ pHash≤12` aynı çifti 22,501 çift/görüntü yükle geri aldı, fakat bu yalnız v2.3
   hipotezidir. Tam tarama başlamayacak.**
7. Uyumlu görüntü modaliteleri/kaynakları arasında sınıflar arası perceptual-duplicate adaylarını
   denetle. Bunu bütün 130.000 dosyanın kör `N²` karşılaştırması yerine doğrulanmış aday-arama
   yöntemiyle yap.
8. Exact ve doğrulanmış yakın-kopyaları aile gruplarına dönüştür.
9. Aileleri bölmeden sabit 70/15/15 manifestleri oluştur ve kesişim testini yap.
10. Ancak bundan sonra adil model eğitimi ve karşılaştırmasına geç.

Bir pilotta bulunan eşik veya kural, yukarıdaki 2–5. adımlarda doğrulanmadan bütün veri setine
uygulanamaz. Buna karşılık, nihai split hazırlanırken 26 alt sınıftan hiçbiri yalnızca pilotta yer
almadığı gerekçesiyle denetimsiz bırakılamaz.
