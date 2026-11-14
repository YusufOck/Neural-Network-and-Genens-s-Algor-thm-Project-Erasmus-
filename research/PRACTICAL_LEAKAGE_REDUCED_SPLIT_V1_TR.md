# Practical Leakage-Reduced Split v1

Tarih: 2026-11-14

Durum: **Üretildi, doğrulandı ve model karşılaştırmasında kullanılmaya hazır**

## Neden bu kararı aldık?

V2.5 ve V2.6b prospective auditleri, metadata'sız 130.000 görüntü içinde bütün doğal yakın
kopyaları sıfır kaçakla bulduğumuzu kanıtlayamadı. V2.6b kuralı 128 outside çiftte 15 same-2D
çifti kaçırdı ve bir çifti belirsiz bıraktı. Yeni eşik/audit döngüleri teslim takvimini tüketirken
split ve model eğitimini sürekli erteliyordu.

Bu nedenle sıfır-kaçak iddiasından vazgeçilmedi; iddia doğru sınırına çekildi. Bu manifest:

- bütün byte-identical SHA-256 gruplarını,
- doğal auditte doğrulanan aynı-kaynak çiftleri,
- V2.5'te aynı-kaynak veya belirsiz bulunan çiftleri,
- V2.6'da doğrulanan same-2D çiftleri,
- V2.6b'de same-2D veya belirsiz bulunan çiftleri

transitif aileler halinde aynı split içinde tutar. Bilinmeyen bütün yakın kopyaların veya aynı
hastaya/seriye ait bütün kesitlerin bulunduğunu iddia etmez.

## Sabit kural

- Hedef: 26 leaf label.
- Seed: 42.
- Oran: %70 train, %15 validation, %15 test.
- Her leaf: tam 3.500 train, 750 validation, 750 test.
- Bir aile parçalanamaz.
- Exact ve doğrulanmış review bağlantıları train/validation/test arasında kesişemez.
- Belirsiz çiftler ihtiyatlı biçimde aynı splitte tutulur. Yanlış pozitif bir aile bağlantısı
  değerlendirmeyi yapay biçimde kolaylaştırmaz; yalnızca iki bağımsız görüntüyü aynı tarafta tutar.

## Doğrulanmış sonuç

| Ölçüm | Sonuç |
|---|---:|
| Toplam görüntü | 130.000 |
| Leaf | 26 |
| Train | 91.000 |
| Validation | 19.500 |
| Test | 19.500 |
| Exact grup | 1.318 |
| Benzersiz review bağlantısı | 85 |
| Toplam family component | 128.560 |
| Çok görüntülü component | 1.402 |
| Çok görüntülü componentlerdeki görüntü | 2.842 |
| En büyük component | 4 |
| Cross-split exact grup | 0 |
| Cross-split review bağlantısı | 0 |
| Cross-split family component | 0 |

Her 26 leaf için dağılım tam 3.500/750/750'dir. Manifest ikinci kez üretildiğinde SHA-256 birebir
aynı kalmıştır:

`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`

## Dosyalar

- Kural: `config/practical_leakage_reduced_split_v1.json`
- Üretici: `src/build_practical_leakage_reduced_split.py`
- Manifest: `research/splits/practical_leakage_reduced_split_v1.csv`
- Makine raporu: `research/splits/practical_leakage_reduced_split_v1_report.json`
- Eğitim manifest yükleyicisi: `src/fixed_split_manifest.py`

Eğitim kodu artık `--split-manifest` parametresiyle bu manifesti yükler ve yeniden rastgele split
üretmez. Manifest kullanılırken `--max-per-class` yasaktır; bütün 130.000 kayıt sabit splitten
gelir.

Örnek 26-sınıf eğitim komutu:

```powershell
py -3.10 .\src\train.py `
  --target-level subclass `
  --model resnet18 `
  --pretrained `
  --split-manifest .\research\splits\practical_leakage_reduced_split_v1.csv `
  --batch-size 16 `
  --epochs 10 `
  --seed 42 `
  --experiment-type final
```

Bu komut henüz çalıştırılmadı. Model eğitimi ayrı kontrollü aşamadır.

## Raporda kullanılacak doğru ifade

Doğru:

> We used a fixed, leaf-stratified, known-family-grouped split that prevents every detected exact
> duplicate and reviewed same-source/conservative pair from crossing partitions.

Yanlış:

> The split is guaranteed free of every near duplicate or patient-level relative.

Veri setinde patient/series ID olmadığı ve prospective retrieval sıfır-kaçak kapısını geçemediği
için ikinci iddia desteklenemez.
