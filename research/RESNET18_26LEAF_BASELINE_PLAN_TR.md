# ResNet18 26-Leaf Baseline — Dondurulmuş Çalıştırma Planı

Tarih: 2026-11-14  
Durum: Notebook ve resume altyapısı hazır; tam eğitim **henüz başlatılmadı**.

## Deney türü ve amacı

Bu koşu ilk tam-kapsam, raporlanabilir `baseline` deneyidir. Amaç, karmaşık fine-tuning veya
genetik algoritmaya geçmeden önce ImageNet-pretrained ResNet18'in sabit split üzerindeki başlangıç
performansını ölçmektir. Bu bir klinik tanı sistemi değildir.

Model 26 leaf etiketinden birini tahmin eder. Tahmin edilen leaf'in dataset içindeki parent grubu
bilindiği için aynı test tahminlerinden ayrıca sekiz parent-group sonucu türetilir. Parent metriği
ayrı bir modelden gelmez.

## Dondurulmuş sözleşme

- Dataset: kabul edilen 130.000 görüntü.
- Split: `practical_leakage_reduced_split_v1.csv`.
- Split SHA-256: `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Train / validation / test: 91.000 / 19.500 / 19.500.
- Model: ResNet18.
- Başlangıç: ImageNet pretrained weights.
- Eğitim: yalnız yeni classifier head; backbone dondurulmuş.
- Girdi: 224×224 RGB.
- Normalization: ImageNet.
- Augmentation: train'de yatay çevirme ve ±5° rotation; validation/test deterministik.
- Optimizer: AdamW.
- Learning rate: `0.001`.
- Weight decay: `0.0001`.
- Batch size: `32`.
- Maksimum epoch: `15`.
- Early stopping: validation macro-F1 dört epoch iyileşmezse durur.
- Scheduler: validation macro-F1 plateau olduğunda learning rate düşürür.
- Seed: `42`.
- Checkpoint seçimi: yalnız en yüksek validation macro-F1.
- Test: seçilen checkpoint yüklendikten sonra değerlendirilir; test metriği model seçmez.

## Colab çalıştırması

Notebook: `notebooks/resnet18_26leaf_baseline_colab.ipynb`

Drive giriş yolu:

`MyDrive/NNGA_DATA/Multi Cancer.zip`

Kalıcı çıktı:

`MyDrive/NNGA_OUTPUTS/resnet18_26leaf_baseline_v1`

Notebook GitHub'ın güncel `main` dalını çeker, dataset ZIP'ini Colab'ın hızlı geçici diskine açar,
CUDA/T4 ortamını ve frozen split hash'ini doğrular. Gerçek eğitim hücresi varsayılan olarak
`AUTHORIZE_FULL_BASELINE = False` ile kilitlidir.

Tahmini süre T4/Colab yoğunluğuna göre yaklaşık 1–3 saattir; bu bir garanti değildir. Her tamamlanan
epoch sonunda Drive'a atomik `last_checkpoint.pth` ve `history.csv` yazılır. Runtime kesilirse aynı
sözleşmeyle notebook yeniden açıldığında son tamamlanan epoch'tan devam edilir; yarım kalan epoch
yeniden çalışabilir. Tamamlanmış bir `test_metrics.json` varsa aynı baseline ikinci kez çalıştırılmaz.

## Beklenen çıktılar

- `best_model.pth`
- `last_checkpoint.pth`
- `history.csv`
- `test_metrics.json`
- `test_predictions.csv` — tam 19.500 satır
- `classification_reports.json`
- leaf ve parent classification report metinleri
- leaf ve parent confusion matrix PNG'leri
- `learning_curves.png`
- frozen kopya `split_manifest.csv`
- Drive'da console log

Başarılı son kontrolde split kopyasının hash'i frozen değerle eşleşmeli, split boyutları
91.000/19.500/19.500 olmalı ve test prediction satırı 19.500 olmalıdır.

## Bu hazırlığın kanıtı

- Notebook JSON'u geçerli ve bütün kod hücreleri Python olarak derleniyor.
- Notebook'ta gerçek eğitim kilitli başlıyor.
- Dondurulmuş parametreler ve beklenen artifact kontrolleri otomatik testlerle doğrulanıyor.
- Yerel kesinti/continue smoke'unda epoch 1 checkpoint'i korunmuş ve `--resume` koşusu doğrudan
  epoch 2'den devam etmiştir. Bu küçük smoke metriği proje performansı değildir.

Tam eğitim çıktısı oluşana kadar accuracy veya macro-F1 için hiçbir baseline sonucu iddia edilmez.
