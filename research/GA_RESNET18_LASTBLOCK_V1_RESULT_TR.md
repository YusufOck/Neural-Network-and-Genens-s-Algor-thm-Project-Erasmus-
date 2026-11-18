# ResNet18 Last-Block GA ve Random Search Sonucu

Tarih: 2026-11-18

Durum: Dondurulmuş Colab/T4 araması tamamlandı. İndirilen review ZIP'i yerelde bağımsız olarak
hash, bütçe, trial sözleşmesi ve test güvenliği bakımından doğrulandı.

Deney türü: `bounded_equal_budget_hyperparameter_search`.

Bu sonuç yalnız proxy validation aramasıdır. Final test performansı, klinik doğrulama veya GA ile
seçilmiş modelin tam-veri sonucu değildir.

## Dondurulmuş sözleşme

- Model: ResNet18; yalnız `layer4 + fc` eğitilebilir.
- Eğitilebilir parametre: 8.407.066.
- Başlangıç: head-only ResNet18 epoch-9 best checkpoint.
- Başlangıç SHA-256:
  `20613069067825ea0dfa30e2242678b863cd0acd7b9be3c5b88f265d39285177`.
- Proxy: her 26 leaf için 350 train + 150 validation; toplam 9.100/3.900 ve test 0.
- Proxy SHA-256:
  `fd6a55def92aa5f3582679ef0be35bde6fcb722be00fb508db9a86a7030aff03`.
- GA: dört aday × üç nesil = 12 benzersiz trial.
- Random search: aynı aralıklardan 12 benzersiz trial.
- Her trial: dört epoch, batch 32, seed 42, aynı initialization ve aynı augmentation.
- Aranan genler: learning rate, weight decay ve dropout.
- Fitness: dört epoch içindeki en yüksek validation macro-F1.
- Test DataLoader'ı oluşturulmaz; test fitness veya seçimde kullanılmaz.

## Ana sonuç

| Yöntem | En iyi aday | Validation macro-F1 | Learning rate | Weight decay | Dropout | En iyi epoch |
|---|---|---:|---:|---:|---:|---:|
| GA | `GA_G01_C02` | **0,952281** | `7,24656e-05` | `1,94541e-05` | `0,461906` | 4 |
| Random search | `RANDOM_002` | 0,949612 | `5,86478e-05` | `3,14775e-04` | `0,282717` | 4 |

Bu tek dondurulmuş koşuda GA'nın en iyi proxy validation sonucu random search'ün en iyisinden
`0,002669`, yani yaklaşık **0,267 yüzde puan** yüksektir.

| Tanımlayıcı ölçüt | GA | Random search | GA - random |
|---|---:|---:|---:|
| En iyi fitness | 0,952281 | 0,949612 | +0,002669 |
| Ortalama fitness | 0,945543 | 0,939380 | +0,006163 |
| Medyan fitness | 0,945231 | 0,938727 | +0,006505 |
| En düşük fitness | 0,937982 | 0,928541 | +0,009441 |

Bu tablo tanımlayıcıdır. Yalnız 12+12 aday ve tek seed olduğu için istatistiksel genelleme ya da
GA'nın her zaman random search'ten üstün olduğu iddiası kurulamaz.

## Nesil yorumu

| GA nesli | Ortalama fitness | En iyi fitness |
|---|---:|---:|
| 1 | 0,945223 | 0,952281 |
| 2 | 0,944424 | 0,952281 |
| 3 | 0,946983 | 0,952281 |

Kazanan `GA_G01_C02` ilk nesilde bulunmuştur. İkinci ve üçüncü nesilde farklı weight-decay
değerleri taşıyan adaylar aynı en iyi fitness değerini yeniden üretmiş, fakat aşmamıştır. Bu nedenle
doğru ifade şudur:

> Bu eşit bütçeli proxy koşusunda GA havuzunun en iyi adayı random-search havuzunun en iyi
> adayından biraz daha yüksek validation macro-F1 verdi.

Şu daha güçlü ifadeler desteklenmez:

- Evrim sonraki nesillerde sürekli iyileştirme sağladı.
- GA genel olarak random search'ten üstündür.
- GA ayarları final test veya klinik performansı artırdı.

## Kazananların dört epoch eğrisi

GA kazananının validation macro-F1 değerleri:

- Epoch 1: 0,915996
- Epoch 2: 0,940071
- Epoch 3: 0,927836
- Epoch 4: 0,952281

Random-search kazananının değerleri:

- Epoch 1: 0,908296
- Epoch 2: 0,940610
- Epoch 3: 0,928645
- Epoch 4: 0,949612

İki kazanan da dördüncü epoch'ta en yüksek değerine ulaşmıştır. Dört epoch'luk proxy hızlı aday
seçimi içindir; uzun eğitimde sıralamanın aynı kalacağı garanti edilmez.

## Bağımsız artifact doğrulaması

- Review ZIP: `ga_resnet18_lastblock_search_v1_review_package.zip`.
- ZIP SHA-256:
  `817c95918dfc997ab75ef8410810d2b86afe44241d72cc4138b160443d804f1b`.
- ZIP girdisi: 126.
- Hash manifestinde kayıtlı artifact: 125; manifestin kendisi hariç eksik/fazla kayıt: 0.
- Hash veya boyut uyuşmazlığı: 0.
- Sonuç satırı: 24; GA 12, random 12.
- Benzersiz candidate ID: 24/24.
- Benzersiz `(learning rate, weight decay, dropout)` üçlüsü: 24/24.
- Dört epoch'u tamamlayan trial: 24/24.
- `test_evaluated=false`: 24/24.
- Split büyüklüğü 9.100/3.900/0: 24/24.
- Başlangıç checkpoint ve proxy hash'i bütün trial'larda aynı: evet.
- Trial summary SHA-256 değerleri sonuç CSV'siyle eşleşiyor: 24/24.
- Fitness değerleri trial özetleriyle eşleşiyor: 24/24.
- `test_metrics.json`: 0.
- Konsol sonucu: `RETURN CODE: 0`.

Konsoldaki DataLoader worker uyarısı sonuç hatası değildir. Çalışma tamamlanmış, bütün trial
kanıtları yazılmış ve final sözleşme kapıları geçmiştir.

## Bilimsel sınırlamalar

- Proxy tam 91.000/19.500 train/validation verisinin küçük, dengeli alt kümesidir.
- Dört epoch yavaş öğrenen fakat uzun vadede daha iyi olabilecek ayarları dezavantajlı kılabilir.
- GA ve random search yalnız bir seed ve 12'şer adayla karşılaştırıldı.
- En iyi GA adayı ilk nesilde bulundu; sonraki nesiller best değeri yükseltmedi.
- Test seti aramada hiç açılmadı. Bu bilimsel olarak doğru olmakla birlikte, burada final model
  doğruluğu olmadığı anlamına gelir.
- Veri setinin modality/source shortcut, önceden augmentation ve bilinmeyen yakın-akraba riskleri
  devam eder.
- Sonuç klinik kanser tanısı, evreleme veya dış veri genellenebilirliği değildir.

## Sonraki kontrollü adım

GA ve random-search kazananları tam 91.000 train / 19.500 validation üzerinde aynı ayrı eğitim
sözleşmesiyle doğrulanmalıdır. İki koşu aynı initialization, maksimum epoch, scheduler, augmentation,
batch, seed ve validation checkpoint kuralını kullanmalıdır. Önce yalnız train/validation ile iki
kazanan seçilip karşılaştırılmalı; ortak test setinin tekrar açılması ayrı gerekçe ve onayla
yapılmalıdır.

Bu tam-veri doğrulama henüz hazırlanmadı veya çalıştırılmadı.
