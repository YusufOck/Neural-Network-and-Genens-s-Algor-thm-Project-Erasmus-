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
