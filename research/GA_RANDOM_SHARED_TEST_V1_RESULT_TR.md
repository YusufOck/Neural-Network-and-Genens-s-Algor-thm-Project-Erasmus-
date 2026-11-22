# GA ve random search shared-test değerlendirmesi v1

Tarih: 2026-11-22

Durum: Tam kapsamlı shared-test değerlendirmesi tamamlandı. İndirilen review ZIP'i, iki
checkpoint-freeze kaydı, 39.000 tahmin satırı, frozen manifest üyeliği, probability alanları ve
metrik aritmetiği yerelde bağımsız olarak doğrulandı.

Deney türü: `shared_benchmark_test_only_evaluation`.

## Sözleşme

İki aday test açılmadan önce tam train/validation sonucuyla seçilmişti:

- GA: `GA_G01_C02`, validation macro-F1 `0,996103985`, epoch 10;
- random search: `RANDOM_002`, validation macro-F1 `0,995640228`, epoch 11.

Shared-test aşamasında model eğitimi, optimizer güncellemesi, hiperparametre değişikliği veya
test-temelli yeniden seçim yapılmadı. Checkpoint hash'leri test görüntüsü açılmadan önce
donduruldu:

- GA checkpoint SHA-256:
  `d5653ff05753e53ff3678d2003d1845cf4994f5799e3b136675333a0389d3cdd`;
- random checkpoint SHA-256:
  `9260abe7717ab65147ac136d02c3ce5b837cf4b4688f8a56bbaa3a27fadb8a1a`.

İki model aynı practical split SHA-256
`999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`
üzerindeki aynı 19.500 test yolunda ölçüldü; her 26 leaf sınıfı 750 görüntü içerir.

## Ana test sonucu

| Yöntem | Leaf accuracy | Leaf macro-F1 | Balanced accuracy | Leaf hata | Parent accuracy | Parent hata |
|---|---:|---:|---:|---:|---:|---:|
| GA | `0,995794872` | `0,995793794` | `0,995794872` | 82 | `0,999846154` | 3 |
| Random search | `0,995743590` | `0,995744525` | `0,995743590` | 83 | `0,999794872` | 4 |

GA eksi random farkı:

- accuracy: `+0,000051282`, yani **+0,00513 yüzde puan**;
- macro-F1: `+0,000049269`, yani **+0,00493 yüzde puan**;
- net doğru görüntü farkı: **1 / 19.500**.

Proxy ve tam-validation sıralaması testte de yön olarak korunmuştur; fakat fark pratik olarak çok
küçüktür.

## Paired karşılaştırma ve McNemar yorumu

Leaf seviyesinde:

- ikisi de doğru: 19.366;
- yalnız GA doğru: 52;
- yalnız random doğru: 51;
- ikisi de yanlış: 31;
- farklı correctness sonucu veren görüntü: 103;
- iki taraflı exact McNemar `p = 1,0`.

Parent seviyesinde yalnız GA doğru 4, yalnız random doğru 3 ve `p = 1,0`'dır.

McNemar, aynı görüntülerde bir modelin diğerinden sistematik olarak daha çok kazanıp kazanmadığını
ölçer. `52` ve `51` neredeyse tam dengeli olduğundan `p=1,0`, GA ile random arasında testte
istatistiksel olarak ayırt edilebilir bir üstünlük kanıtı bulunmadığını gösterir. Bu değer
“modeller tamamen aynıdır” anlamına gelmez: 103 görüntüde doğruluk durumları ayrışmıştır, fakat
net yön yalnız bir görüntüdür.

İkisinin de yanlış olduğu 31 görüntünün tamamında aynı yanlış leaf tahmin edilmiştir. Bu ortak zor
kümenin gerçek sınıf dağılımı: oral normal 12, oral SCC 10, breast benign 5, lung adenocarcinoma
2, brain meningioma 1 ve breast malignant 1.

## Sınıf bazlı davranış

GA'nın random'a göre en büyük F1 kazanımları:

- `lymph_mcl`: `+0,00673`;
- `lymph_fl`: `+0,00397`;
- `colon_bnt`: `+0,00267`;
- `lymph_cll`: `+0,00266`.

Random'ın GA'ya göre en büyük F1 kazanımları:

- `oral_scc`: `+0,00767`;
- `oral_normal`: `+0,00510`;
- `breast_malignant`: `+0,00265`;
- `brain_tumor`: `+0,00201`.

GA'nın 82 hatasının 53'ü oral normal/SCC arasındadır: `oral_scc -> oral_normal` 31 ve
`oral_normal -> oral_scc` 22. Random'da bu iki yön toplam 42 hatadır. Buna karşılık random,
`lymph_mcl` için 12 hata üretirken GA yalnız iki `lymph_mcl -> lymph_fl` hatası üretmiştir.
Makro ortalamada bu farklı güçlü/zayıf bölgeler birbirini büyük ölçüde dengelemiştir.

## Loss ve confidence

Random test loss `0,0125813`, GA test loss `0,0135831` ile biraz daha düşüktür. Accuracy/F1 yalnız
doğru sınıfı sayarken cross-entropy loss bütün probability dağılımını ve yanlış tahmin güvenini
de dikkate alır. GA'nın bir görüntü fazla doğru bilmesine rağmen daha yüksek loss üretmesi bu
nedenle çelişki değildir.

- GA yanlışlarının ortalama confidence değeri `0,81595`; 36 yanlış `>=0,90`, 13 yanlış
  `>=0,99` confidence taşır.
- Random yanlışlarının ortalaması `0,80539`; 32 yanlış `>=0,90`, 11 yanlış `>=0,99` confidence
  taşır.

Bu softmax değerleri klinik güven veya doğruluk garantisi değildir.

## Önceki elle seçilmiş last-block baseline ile karşılaştırma

Önceki kontrollü last-block ResNet18 sonucu aynı shared testte leaf accuracy/macro-F1
`0,996462/0,996462` ve 69 leaf hatası üretmişti.

- GA, bu baseline'dan accuracy bakımından yaklaşık `-0,0667` yüzde puan ve macro-F1 bakımından
  `-0,0669` yüzde puan düşüktür; 13 ek hata üretmiştir.
- Random, baseline'dan accuracy bakımından yaklaşık `-0,0718` yüzde puan ve macro-F1 bakımından
  `-0,0718` yüzde puan düşüktür; 14 ek hata üretmiştir.

Dolayısıyla GA, eşit bütçeli random adayı validation ve testte çok küçük farkla geçmiştir; fakat
elle seçilmiş önceki hiperparametrelerin shared-test sonucunu iyileştirmemiştir. Bu başarısız bir
proje sonucu değildir: GA'nın nasıl sınırlandırıldığı, eşit bütçeli random ile nasıl
karşılaştırıldığı ve küçük farkın neden abartılmaması gerektiği projenin deneysel katkısıdır.

## Artifact ve veri bütünlüğü

- Review ZIP SHA-256:
  `03ad9a66b089456175923ffc1bb24d011ea987dd908524ed86688aba4f4efd20`.
- ZIP girdisi: 21.
- Review manifest: 20/20 artifact, hash/boyut uyuşmazlığı 0.
- İç run hash manifesti: 18/18, uyuşmazlık 0.
- Her aday: 19.500 satır ve 19.500 benzersiz relative path.
- GA/random path kümeleri ve true label'ları birebir aynı.
- Frozen test manifestine eksik/fazladan yol: 0.
- True leaf/parent metadata uyuşmazlığı: 0.
- Probability anahtar/toplam, argmax ve confidence uyuşmazlığı: 0.
- JSON metrikleri tahmin CSV'lerinden bağımsız yeniden hesaplandı ve eşleşti.
- Console log `RETURN CODE: 0` ile tamamlandı.

Colab, `num_workers=4` için ortamın önerdiği worker sayısının 2 olduğuna dair performans uyarısı
vermiştir. Run tamamlandığı ve bütün çıktılar doğrulandığı için bu doğruluk hatası değildir; yeniden
çalıştırma gerektirmez.

## Savunulabilir sonuç

Şu ifade desteklenir:

> Eşit bütçeli aramada GA tarafından seçilen aday, random-search adayını tam validation'da ve aynı
> shared testte yön olarak geçti; fakat test farkı yalnız bir görüntü ve exact McNemar p-değeri
> 1,0 olduğu için anlamlı bir üstünlük gösterilmedi.

Şu ifadeler desteklenmez:

- genetik algoritma random search'ten genel olarak üstündür;
- GA nihai model performansını elle seçilmiş baseline'a göre artırmıştır;
- yüzde 99,58 sonucu klinik kanser tanı doğruluğudur;
- sekiz-parent skorları kanser var/yok veya evre doğruluğudur;
- shared test dokunulmamış dış doğrulamadır.

## Sonraki aşama

GA/random deney zinciri artık tamamlanmıştır: proxy search, tam-validation confirmation ve frozen
shared-test karşılaştırması vardır. Yeni hiperparametre araması başlatmak yerine mevcut kanıtlar
İngilizce nihai rapor, ana karşılaştırma tablosu ve sunuma dönüştürülmelidir.

