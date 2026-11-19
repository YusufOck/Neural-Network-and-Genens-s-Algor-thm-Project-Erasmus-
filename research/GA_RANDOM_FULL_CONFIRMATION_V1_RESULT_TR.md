# GA ve random search tam-veri validation doğrulaması v1

Tarih: 2026-11-19

## Sonucun kapsamı

Bu çalışma, proxy aramada seçilen bir GA adayı ile eşit bütçeli random search adayını aynı
tam train/validation sözleşmesinde karşılaştırır. Çalışma 91.000 train ve 19.500 validation
görüntüsü kullanmıştır; test görüntüsü kullanılmamış, test loader kurulmamış ve test metriği
üretilmemiştir.

Ortak koşullar:

- model: pretrained ResNet18;
- eğitilen bölüm: yalnız `layer4 + fc`;
- başlangıç checkpoint'i: aynı epoch-9 frozen-head checkpoint'i;
- seed: 42;
- batch size: 32;
- en fazla 12 epoch ve patience 4;
- checkpoint seçimi: validation macro-F1;
- 26 alt sınıfın her birinde 3.500 train ve 750 validation görüntüsü;
- test satırı: 0.

Bu nedenle aşağıdaki sayılar test başarısı veya klinik doğrulama değildir.

## Doğrulanmış sonuçlar

| Yöntem | Aday | Learning rate | Weight decay | Dropout | En iyi epoch | Full-validation macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Genetik algoritma | `GA_G01_C02` | `7,246558e-05` | `1,945407e-05` | `0,461906` | 10 | `0,996103985` |
| Random search | `RANDOM_002` | `5,864780e-05` | `3,147753e-04` | `0,282717` | 11 | `0,995640228` |

Önceden tamamlanan elle seçilmiş last-block baseline'ın en iyi validation macro-F1 değeri
`0,995948075` idi.

Farklar:

- GA eksi random search: `+0,000463757`, yani yaklaşık **+0,0464 yüzde puan**;
- GA eksi elle seçilmiş baseline: `+0,000155910`, yani yaklaşık **+0,0156 yüzde puan**;
- random search eksi elle seçilmiş baseline: `-0,000307847`, yani yaklaşık **-0,0308 yüzde puan**.

Proxy aşamasındaki sıralama tam validation'da da korunmuştur: GA adayı random adayından yüksek
çıkmıştır. Bununla birlikte fark çok küçüktür. Tek seed ve yalnız bir GA/random kazanan çiftiyle
bu fark, genetik algoritmanın genel olarak üstün olduğunu kanıtlamaz. Proxy GA kazananının daha
ilk nesilde bulunmuş olması da sonraki nesillerde düzenli evrimsel iyileşme iddiasını desteklemez.

## Eğitim eğrisi kontrolü

Her iki aday da 12 epoch tamamlamıştır.

- GA validation macro-F1 değeri epoch 8'de `0,995230202`, epoch 10'da en yüksek
  `0,996103985`, epoch 11'de `0,995023074` ve epoch 12'de `0,995537727` olmuştur.
- Random validation macro-F1 değeri epoch 8'de `0,994720972`, epoch 11'de en yüksek
  `0,995640228` ve epoch 12'de `0,995589745` olmuştur.

Son epoch yerine validation macro-F1 ile seçilen en iyi checkpoint'in korunması doğrudur. Küçük
son-epoch dalgalanmaları yarım veya bozuk çalışmaya işaret etmez.

## Artifact bütünlüğü

İndirilen inceleme paketi:

- dosya: `ga_random_full_confirmation_v1_review_package.zip`;
- SHA-256: `01bf6413736f590084e21c099801910d637b549ebd7852b0f560b4efdde0a8bb`;
- ZIP girdisi: 18;
- manifestte doğrulanan artifact: 17/17;
- hash veya boyut uyuşmazlığı: 0;
- test artifact'i: 0;
- durum: `ga_random_full_confirmation_v1_complete`;
- `test_evaluated`: `false`.

Run contract; config, runner, trainer, split, confirmation manifesti, başlangıç checkpoint'i ve
kaynak proxy-search summary hash'lerini kilitlemiştir. İki beklenen adayın ikisi de tamamlanmıştır.

## Bilimsel yorum

Bu deney şu iddiayı destekler:

> Eşit proxy bütçesiyle bulunan iki aday aynı tam validation sözleşmesinde yeniden eğitildiğinde,
> GA tarafından seçilen aday validation macro-F1 bakımından küçük bir farkla daha yüksek sonuç
> verdi.

Şu iddiaları desteklemez:

- GA her zaman random search'ten üstündür;
- fark farklı seed'lerde aynı kalır;
- sonuç bağımsız veya klinik veride geçerlidir;
- test başarısı `0,9961`'dir;
- GA arama süreci nesilden nesle düzenli iyileşmiştir.

## Sonraki kontrollü karar

Validation kazananı artık `GA_G01_C02` olarak dondurulabilir. Herhangi bir test değerlendirmesi,
bu sonuçtan ayrı bir sözleşme ve açık onay gerektirir. Mevcut test kümesi önceki model
deneylerinde kullanıldığı için ileride yapılacak değerlendirme “untouched final test” değil,
aynı shared benchmark üzerinde ek karşılaştırma olarak raporlanmalıdır.

