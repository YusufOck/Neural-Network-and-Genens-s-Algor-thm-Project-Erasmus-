# ResNet18 Last-Block Fine-Tuning Kontrollü Ablation Planı

Tarih: 2026-11-17  
Durum: **Kod, kilitli Colab notebook'u ve doğrulama testleri hazır; tam eğitim başlatılmadı.**

## Araştırma sorusu

İlk ResNet18 deneyinde ImageNet gövdesinin tamamı donuktu ve yalnız son 26-sınıflı başlık
eğitildi. Aynı frozen splitte SmallCNN daha yüksek test macro-F1 üretti. Bu deney şu dar soruyu
yanıtlayacak:

> ResNet18'in son özellik bloğunu (`layer4`) veri setine uyarlamak, head-only ResNet18'in
> validation ve test performansını iyileştiriyor mu?

Bu bir mimari üstünlük veya klinik geçerlilik testi değildir. Test seti önceki iki modelde zaten
görüldüğü için sonuç `controlled ablation on the shared benchmark` olarak adlandırılacaktır.
Model/hyperparameter seçimi yine yalnız validation macro-F1 ile yapılacaktır.

## Başlangıç ve eğitim kapsamı

- Mimari: ResNet18.
- Başlangıç ağırlığı: tamamlanmış head-only baseline'ın validation ile seçilen epoch-9
  `best_model.pth` dosyası.
- Başlangıç checkpoint SHA-256:
  `20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177`.
- Eğitilebilir katmanlar: yalnız `layer4` ve yeni classifier `fc`.
- Donuk katmanlar: stem, `layer1`, `layer2`, `layer3`.
- Eğitilebilir parametre: 8.407.066.
- Eski optimizer, scheduler ve history **taşınmaz**. Bu nedenle `--resume` değil,
  `--init-checkpoint` yeni deney başlangıcıdır.
- Yeni koşu kesilirse ancak o zaman yeni output içindeki `last_checkpoint.pth` ile `--resume`
  kullanılır.

## Dondurulmuş deney sözleşmesi

- Hedef: 26 leaf sınıf; aynı tahminlerden sekiz parent-group metriği.
- Split: `practical_leakage_reduced_split_v1.csv`.
- Split SHA-256:
  `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Boyutlar: 91.000 train, 19.500 validation, 19.500 test.
- Görüntü: 224×224, ImageNet normalization, minimal train-only augmentation.
- Optimizer: AdamW.
- Learning rate: `0.0001`.
- Weight decay: `0.0001`.
- Batch size: 32.
- Seed: 42.
- En fazla epoch: 12.
- Early stopping: validation macro-F1 dört epoch iyileşmezse dur.
- Checkpoint seçimi: yalnız en yüksek validation macro-F1.
- Deney tipi: `ablation`.
- Ayrı output: `MyDrive/NNGA_OUTPUTS/resnet18_lastblock_finetune_v1`.

## Kod güvenliği

`src/train.py` artık iki işlemi açıkça ayırır:

1. `--init-checkpoint`: model ağırlıklarını önceki en iyi checkpointten yükler, fakat optimizer,
   scheduler ve history'yi sıfırdan kurar.
2. `--resume`: aynı output klasöründeki aynı training contract'a ait son epoch durumunu geri
   yükler.

Init checkpoint yüklenmeden önce dosya hash'i, mimari, target level ve 26 sınıf sırası doğrulanır.
Init checkpoint hash'i yeni training contract'a eklenir; bu nedenle farklı başlangıç ağırlığıyla
yanlış resume reddedilir. Test metrikleri ayrıca init provenance ve trainable/total parametre
sayılarını kaydeder.

## Colab kullanım sırası

Notebook: `notebooks/resnet18_lastblock_finetune_colab.ipynb`

1. T4 GPU seçilir.
2. Drive bağlanır; repository, ZIP, frozen split ve eski best checkpoint hash'leri doğrulanır.
3. Eğitim yapmayan ön kontrol yalnız `layer4 + fc` katmanlarının açık ve trainable parametre
   sayısının 8.407.066 olduğunu doğrular.
4. Tam eğitim hücresi başlangıçta
   `AUTHORIZE_FULL_FINETUNE = False` ile kilitlidir.
5. Ayrı çalıştırma onayından sonra `True` yapılır ve tam koşu başlar.
6. Kesinti olursa aynı notebook/hücre mevcut yeni checkpointten devam eder; eski baseline'a
   yazmaz.
7. Tamamlanınca 19.500 tahmin, split/init hash'i ve final artifact kapsamı doğrulanır; checkpoint
   dosyalarını içermeyen review ZIP'i oluşturulur.

## Kaynak maliyeti ve beklenen çıktı

T4 üzerinde yaklaşık 2–4 saat öngörülür; bu garanti değildir. Head-only eğitimden daha yavaştır,
çünkü 8,4 milyon parametre için gradient hesaplanır. Çıktı accuracy, macro-F1, balanced accuracy,
per-class rapor, leaf/parent confusion matrix, learning curves ve görüntü-bazlı test tahminlerini
içerir.

Bu hazırlık bir performans sonucu değildir. Tam run ve sonuç yorumu ayrı checkpointlerdir.
