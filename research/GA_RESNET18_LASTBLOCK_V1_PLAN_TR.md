# ResNet18 Last-Block Genetik Algoritma ve Random Search Planı

Tarih: 2026-11-18

Durum: Deney sözleşmesi, proxy manifest, restart-safe çalıştırıcı ve kilitli Colab notebook'u
donduruldu; 226/226 proje testi geçti. Hiçbir search trial çalıştırılmadı ve test seti açılmadı.

Deney türü: `bounded_equal_budget_hyperparameter_search`

## 1. Amaç

Amaç, tamamlanmış ResNet18 `layer4 + fc` fine-tuning yapısını değiştirmeden üç eğitim
hiperparametresini genetik algoritma ile aramak ve sonucu aynı hesaplama bütçeli random search ile
karşılaştırmaktır. Bu aşamanın bilimsel sorusu şudur:

> Aynı arama uzayı, aynı veri proxy'si ve aynı sayıda model eğitimi altında genetik algoritma,
> bağımsız rastgele örneklemeden daha yüksek validation macro-F1 üreten bir tarif buluyor mu?

GA kullanmak tek başına başarı değildir. Adil random-search karşılaştırması yapılmadan GA'nın
faydalı olduğu söylenmeyecektir.

## 2. Genetik algoritmanın projedeki karşılığı

- **Birey:** Bir eğitim tarifi.
- **Genler:** Learning rate, weight decay ve dropout.
- **Population:** Aynı round'da değerlendirilen dört tarif.
- **Fitness:** Dört proxy epoch içindeki en yüksek validation macro-F1.
- **Selection:** İki adaylı tournament seçimi.
- **Crossover:** İki ebeveynden her genin bağımsız seçilmesi.
- **Mutation:** Her genin `%35` olasılıkla kendi aralığında küçük değişmesi.
- **Round/generation:** Dört yeni adaylık bir değerlendirme turu.

GA, tek bir modeli evrimleştirmez. Her birey aynı başlangıç checkpointinden başlayan ayrı, kısa bir
eğitim koşusudur.

## 3. Sabit model sözleşmesi

- Model: ImageNet-pretrained ResNet18.
- Eğitilebilir bölüm: yalnız `layer4 + fc`.
- Başlangıç checkpointi: tamamlanmış head-only ResNet18 epoch-9 best.
- Init checkpoint SHA-256:
  `20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177`
- Hedef: 26 leaf sınıf.
- Görüntü: 224×224, ImageNet normalizasyonu.
- Augmentation: minimal horizontal flip + ±5 derece rotation.
- Optimizer: AdamW.
- Batch size: 32.
- Seed: 42.
- Aday başına epoch: 4.
- Early stopping: arama trial'larında kapalı; herkes aynı dört epoch'u alır.

Fine-tuning kapsamı gen yapılmamıştır. Head-only ile layer4 gibi farklı kapsamlar farklı hesaplama
maliyeti ve farklı parametre sayısı taşır; bunları aynı trial bütçesinde karıştırmak adil olmaz.

## 4. Arama uzayı

| Gen | Aralık | Örnekleme |
|---|---:|---|
| Learning rate | `1e-5 – 3e-4` | log10-uniform |
| Weight decay | `1e-6 – 1e-3` | log10-uniform |
| Dropout | `0,10 – 0,50` | linear-uniform |

Logaritmik örnekleme, learning rate ve weight decay gibi mertebe düzeyinde değişen değerlerin
küçük bölgede yığılmasını önler.

## 5. Eşit bütçe

- GA: `4 population × 3 round = 12` benzersiz aday.
- Random search: aynı aralıklardan `12` benzersiz aday.
- Toplam: 24 kısa trial.
- Her trial: aynı proxy, aynı dört epoch, aynı initialization ve seed.
- GA'da iyi bireyler arşivde korunur fakat tekrar eğitilerek bütçe tüketmez.

Karşılaştırmanın ana çıktısı her yöntemin 12 trial içindeki en yüksek validation macro-F1 değeridir.
Trial sayıları eşit değilse deney tamamlanmış sayılmaz.

## 6. Proxy manifest

Kaynak, değiştirilmeyen practical split'tir:

`research/splits/practical_leakage_reduced_split_v1.csv`

Kaynak SHA-256:

`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`

Proxy:

`research/splits/ga_search_proxy_v1.csv`

Proxy SHA-256:

`fd6a55def92aa5f3582679ef0be35bde6fcb722be00fb508db9a86a7030aff03`

Seçim, her original split/leaf içinde `seed + split + leaf + relative_path` SHA-256 sırasıyla
yapılmıştır:

- Her leaf: 350 train + 150 validation.
- Toplam: 9.100 train + 3.900 validation.
- Test: 0.
- Leaf sayısı: 26.

Proxy train yolları yalnız ana train bölümünden, proxy validation yolları yalnız ana validation
bölümünden gelir. Bir yolun original partition'ı değiştirilemez. Test yolu proxy'ye girerse trainer
işlemi durdurur.

## 7. Test güvenliği

Arama sırasında:

- Test DataLoader oluşturulmaz.
- Test görüntüsü açılmaz.
- `test_metrics.json` üretilmez.
- Fitness veya seçimde test metriği kullanılmaz.
- Trial özeti açıkça `test_evaluated: false` taşır.

GA ve random-search kazananları yalnız proxy validation sonucuyla seçilecektir. İki kazananın tam
train/validation eğitimi ve daha sonra test değerlendirmesi ayrı plan, ayrı onay ve ayrı Colab
çalışması olacaktır.

## 8. Kesinti ve devam

Her aday kendi klasörüne yazılır. Bir trial kesilirse aynı adayın `last_checkpoint.pth` dosyasıyla
devam edilir. Tamamlanmış trial yeniden eğitilmez. Aday parametreleri eğitimden önce
`candidate.json` içine yazılır; resume sırasında değiştirilirse işlem durur.

Üst düzey kanıtlar:

- `run_contract.json`
- `frozen_config.json`
- `search_results.csv`
- `search_summary.json`
- Her trial için `candidate.json`, `search_trial_summary.json`, `trial_result.json`, `history.csv`
  ve console log.

## 9. Kabul koşulları

- Frozen split, proxy ve init checkpoint hash'leri eşleşmeli.
- GA 12/12 ve random search 12/12 tamamlanmalı.
- Her yöntem içinde adaylar benzersiz ve bütün genler sınırlar içinde olmalı.
- Bütün trial özetlerinde `test_evaluated=false` olmalı.
- Her trial tam dört epoch kullanmalı.
- İki yöntem aynı proxy, initialization, seed, batch ve augmentation sözleşmesini kullanmalı.
- Kazanan seçimi yalnız validation macro-F1 ile yapılmalı.

## 10. Sınırlılıklar

- Dört proxy epoch, hızlı öğrenen ayarları yavaş fakat sonunda daha iyi olabilecek ayarlara tercih
  edebilir.
- Proxy dengeli ve deterministik olsa da tam train/validation setinden küçüktür.
- 12 trial küçük bir GA bütçesidir; bütün arama uzayının optimumunu garanti etmez.
- Önceki deneyler ortak test setini kullanmıştır; ilerideki sonuç dış veri doğrulaması değildir.
- Yüksek veri-seti başarısı klinik geçerlilik anlamına gelmez.

## 11. Çalıştırma kilidi

Kod ve Colab notebook'u hazırlandıktan sonra yalnız preflight çalıştırılacaktır. Uzun arama,
Mehmet'in frozen hash'leri, 9.100/3.900/0 sayımını ve `12 + 12` bütçeyi görüp ayrı onay vermesiyle
başlayacaktır. Bu planın hazırlanması search trial yetkisi değildir.

## 12. Dondurulmuş uygulama hash'leri

- Trainer: `807a23e40ea3aaa5c0c98737beddb637d71c5e5a9e5f230aa0b75d0be7212a77`
- GA/random runner: `657190d6b61ec1a1ce7a7aba6b5345e4fbcee376431a1ea9825d3b1dc4c1df7d`
- Proxy builder: `cf4fb5b0825b076b9c23414d5dcc342288a06e64430729cccec20d7d26d60fc2`
- Frozen config: `fefe845a8fb84c0f252954f42cee3a8f4fa84210a4313d24af339fdeb90c7638`
- Proxy manifest: `fd6a55def92aa5f3582679ef0be35bde6fcb722be00fb508db9a86a7030aff03`
- Proxy report: `564645c77b1e1e07d56b3015f0e5eaf06e96fd1a108cb7af111def18a1bf2dfc`
- Colab notebook: `a3eddf9c3d39daf816a548ee0b0d2e57f709041d78b604548e549c4d78071bea`
