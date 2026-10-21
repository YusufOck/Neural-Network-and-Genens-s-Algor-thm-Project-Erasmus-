# NNGA Multi-Cancer Classification Project

Bu proje, Bükreş Politeknik Üniversitesi Neural Networks and Genetic Algorithms dersi için
Kaggle'daki **Multi Cancer Dataset** üzerinde yürütülen görüntü sınıflandırma çalışmasıdır.

Profesör kayıtta sekiz sınıf istemiştir; ancak veri setinin sekiz üst klasörü sekiz tutarlı
kanser tanısı değil, farklı kaynak/organ gruplarıdır. Bazılarının içinde hem normal/benign hem
de hastalıklı görüntüler bulunur. Bu nedenle çalışma iki seviyeyi birlikte ele almayı planlar:

1. Profesörün sekiz sınıf ifadesiyle uyumlu üst grup değerlendirmesi.
2. Veri setindeki daha ayrıntılı 26 alt sınıfın tahmini ve buradan sekiz üst grubun türetilmesi.

Bu kapsam henüz nihai deney sonucu veya klinik tanı iddiası değildir.

## Temel belgeler

- [PROJECT_PLAN.md](PROJECT_PLAN.md): deney sırası, başarı ölçütleri ve teslim planı
- [research/PROJECT_STATUS_TR.md](research/PROJECT_STATUS_TR.md): tam kapsam, pilot, devam eden ve başlamamış işlerin güncel kontrol listesi
- [research/NEAR_DUPLICATE_METHOD_PLAN_TR.md](research/NEAR_DUPLICATE_METHOD_PLAN_TR.md): pHash, ORB/RANSAC, ECC ve hizalanmış benzerlikle bütün 26 sınıfa genişleme planı
- [research/CALIBRATION_SET_V1_AUDIT_TR.md](research/CALIBRATION_SET_V1_AUDIT_TR.md): 26 sınıflı duplicate-yöntem kalibrasyon taslağının üretim ve kontrol sonuçları
- [research/CALIBRATION_REVIEW_PACKAGE_TR.md](research/CALIBRATION_REVIEW_PACKAGE_TR.md): 278 aday için Pair ID'li temas sayfaları ve kontrollü development/holdout inceleme protokolü
- [research/DATASET_MASTER_AUDIT_TR.md](research/DATASET_MASTER_AUDIT_TR.md): veri setinin ana denetimi
- [research/LABEL_SEMANTICS_AUDIT_TR.md](research/LABEL_SEMANTICS_AUDIT_TR.md): 26 etiketin anlamları ve sınırları
- [research/EXISTING_WORKS.md](research/EXISTING_WORKS.md): Kaggle/GitHub çalışmalarının karşılaştırması
- [research/ROUPB_CANCER_APP_AUDIT_TR.md](research/ROUPB_CANCER_APP_AUDIT_TR.md): en yakın 26-sınıf dış projenin kod denetimi

## Öğretmenin istediği teslim

- Kayıttaki ifadeye göre sekiz sınıflı görüntü sınıflandırma yazılımı
- Yaklaşık 15-20 sayfalık teorik ve deneysel rapor
- Proje sunumu/savunması
- Kullanılan dış kodların kaynaklandırılması ve öğrencinin katkısının açıklanması

## Veri setinin önemli sınırları

Sekiz üst grup şunlardır:

1. ALL
2. Brain Cancer
3. Breast Cancer
4. Cervical Cancer
5. Kidney Cancer
6. Lung and Colon Cancer
7. Lymphoma
8. Oral Cancer

> Bu sekiz etiket sağlıklı/kanserli veya kanser evresi etiketi değildir. Örneğin
> `oral_normal` üst seviyede `Oral Cancer`, `kidney_normal` ise `Kidney Cancer` grubuna girer.
> Veri seti MRI, CT, kan yayması, Pap smear ve farklı histopatoloji görüntülerini bir araya
> getirir. Model hastalık özelliği yerine görüntüleme yöntemini öğrenebilir. Çalışma klinik tanı
> sistemi olarak sunulmamalıdır.

Yerel veri yolu:

```text
archive/Multi Cancer/Multi Cancer
```

Kaynak veri GitHub'a yüklenmez ve kod tarafından değiştirilmez. İki `*.bak.jpg` dosyası deney
indeksinden çıkarıldığında 26 alt sınıfta toplam 130.000 kabul edilen görüntü bulunur.

## Yerel ortam

Aktif proje sürümü **Python 3.10.11**'dir. 19 Ekim 2026 tarihinde doğrulanan temel ortam:

- PyTorch `2.5.1+cu118`
- torchvision `0.20.1+cu118`
- NumPy `2.2.6`
- Pillow `12.3.0`
- CUDA kullanılabilir

Python 3.10 ortamında `scikit-learn` ve `matplotlib` henüz kurulmamıştır. pHash pilotu yalnızca
NumPy ve Pillow kullanır; model eğitimi öncesinde eksik eğitim bağımlılıkları ayrıca kurulup
doğrulanacaktır.

## Tamamlanan veri denetimleri

Temel veri denetimi:

```powershell
py -3.10 src/audit_dataset.py
```

Tüm görüntüleri açarak bozuk dosya kontrolü:

```powershell
py -3.10 src/audit_dataset.py --verify-all
```

SHA-256 ile byte-byte aynı dosyaları bulma:

```powershell
py -3.10 src/audit_exact_duplicates.py
```

SHA-256 denetimi 1.318 birebir kopya grubu ve bu gruplarda 2.673 dosya bulmuştur. SHA-256
yalnızca dosya içeriği tamamen aynı olan kopyaları belirler; döndürülmüş, parlaklığı değiştirilmiş
veya yeniden kodlanmış akrabaları bulamaz.

## Tamamlanan kontrollü adım: pHash pilotu

`src/audit_perceptual_duplicates.py`, seçilen alt sınıflardaki görüntüler için 64-bit pHash
hesaplar ve Hamming uzaklığı aralıklarından manuel inceleme örnekleri üretir. Kod otomatik eşik
seçmez, split oluşturmaz ve model eğitmez.

2026-10-19 tarihinde Python 3.10 ile çalıştırılan pilot komutu:

```powershell
py -3.10 src/audit_perceptual_duplicates.py `
  --subclasses lung_scc brain_glioma `
  --samples-per-band 12 `
  --seed 42 `
  --workers 4
```

Pilot 10.000 görüntüyü ve iki sınıfta toplam 24.995.000 çifti taradı; 120 örneği 20 temas
sayfasında incelemeye açtı. Bilinen 279 SHA-256 tam-kopya çiftinin tamamı pHash uzaklığı 0 verdi.
Ancak yöntem görüntü türüne çok farklı tepki verdi: `brain_glioma` için 12 veya daha küçük
uzaklıkta 173.992 aday, `lung_scc` için yalnız 1.731 aday oluştu. Görsel denetimde beyin MR
adaylarının çoğu ortak arka plan/anatomi kaynaklı false positive iken akciğer histoloji örnekleri
aynı mikroskobik alanın dönüşümleri görünümündeydi. Bu nedenle henüz otomatik/global eşik
seçilmemiştir ve split oluşturulmamıştır. Ayrıntılı sonuç:
[`research/PHASH_PILOT_AUDIT_TR.md`](research/PHASH_PILOT_AUDIT_TR.md).

Manuel kararları sohbetten bağımsız ve makine-okunur biçimde saklamak için mevcut 120 adaydan
güvenli bir `review_labels.csv` şablonu hazırlanabilir:

```powershell
py -3.10 src/prepare_phash_review.py
```

Bu yardımcı araç görüntüleri sınıflandırmaz. SHA-256 ile zaten byte-byte aynı olduğu kanıtlanan
çiftleri önceden işaretler; diğer satırları `same_source_candidate`, `independent_lookalike` veya
`uncertain` görsel kararlarından biri girilmek üzere boş bırakır. Var olan tabloyu normal
çalıştırmada ezmez. Dosya yerel bir audit çıktısı olduğu için `artifacts/` altında tutulur ve
GitHub'a gönderilmez. Bu 120 satır yalnızca iki alt sınıftaki eski pHash pilotunun kaydıdır;
bütün 26 alt sınıfı kapsayacak resmi kalibrasyon kümesinin yerine geçmez.

Temas sayfaları, `candidate_samples.csv` ve `review_labels.csv` aynı `P0001`–`P0120` Pair ID
değerlerini kullanır. Eski bir pilot çıktısına Pair ID eklemek için 24.995.000 çifti yeniden
taramadan yalnızca mevcut adaylar tekrar çizilebilir:

```powershell
py -3.10 src/audit_perceptual_duplicates.py --rerender-existing
```

### Bütün 26 sınıflı duplicate kalibrasyon taslağı

Eski iki-sınıflı pilotun yerine geçmek üzere, bütün 26 alt sınıftan kontrollü pozitif ve insan
incelemeli adaylar şu komutla üretildi:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\build_duplicate_calibration_set.py --seed 42 --workers 4
```

Bu komut model eğitmez ve 70/15/15 sınıflandırma split'i oluşturmaz. Üretilen audit artefaktları
`artifacts/audit/duplicate_calibration_v1/` altında yerel tutulur. Doğrulanmış sayılar ve pHash
recall bulguları için
[`research/CALIBRATION_SET_V1_AUDIT_TR.md`](research/CALIBRATION_SET_V1_AUDIT_TR.md) dosyasına bakın.

278 belirsiz adayın Pair ID'li temas sayfalarını ve güvenli inceleme tablosunu üretmek için:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\render_calibration_review.py --pairs-per-page 6 --thumb-size 320
```

Önce yalnız development çiftleri etiketlenecek; holdout, yöntem dondurulana kadar kullanılmayacak.
İnceleme paketi ve karar sözlüğü
[`research/CALIBRATION_REVIEW_PACKAGE_TR.md`](research/CALIBRATION_REVIEW_PACKAGE_TR.md) içinde
açıklanmıştır.

## Eğitim kodunun mevcut durumu

`src/train.py` şu modelleri destekleyen bir temel uygulamadır:

- SmallCNN
- ResNet18
- DenseNet121
- EfficientNet-B0

Mevcut kod küçük smoke test/pilotlar çalıştırabilmiştir; ancak henüz nihai deney için hazır
değildir. Sabit ve kopya-farkındalıklı split manifesti oluşturulmadan yeni model karşılaştırması
başlatılmayacaktır. Nihai değerlendirmede accuracy yanında macro-F1, balanced accuracy, sınıf
başına recall ve confusion matrix kullanılacaktır.

## GitHub kapsamı

Depoya kod, yapılandırma ve araştırma notları gönderilir. Aşağıdakiler yerelde tutulur:

- `archive/`: yaklaşık 8,6 GiB kaynak veri seti
- `artifacts/`: yeniden üretilebilir çalışma çıktıları ve görsel örnekler
- `transcript.pdf` ve ham ses/video kayıtları
- model ağırlıkları (`*.pt`, `*.pth`, `*.ckpt`) ve dışarı aktarılan modeller (`*.onnx`)
- üçüncü taraf depoların yerel kopyaları (`research/external/`)

Anlamlı ve doğrulanmış her geliştirme açıklayıcı bir Git commit'i olarak `main` dalına gönderilir.
Bozuk, yarım veya henüz doğrulanmamış deney sonuçları başarı sonucu gibi paylaşılmaz.
