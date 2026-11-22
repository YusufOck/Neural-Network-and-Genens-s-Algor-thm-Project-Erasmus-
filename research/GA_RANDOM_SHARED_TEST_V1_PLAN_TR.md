# GA ve random search kazananları shared-test değerlendirme planı v1

Tarih: 2026-11-22

Durum: Uygulama sözleşmesi, iki-aşamalı runner ve kilitli Colab notebook'u hazır. 245/245
proje testi geçti; test görüntüsü henüz açılmadı.

Deney türü: `shared_benchmark_test_only_evaluation`.

## Amaç

Tam train/validation doğrulamasında seçilmiş `GA_G01_C02` ve `RANDOM_002` checkpoint'lerini
yeniden eğitmeden aynı 19.500 test görüntüsünde karşılaştırmak. İki modelin hiperparametreleri,
epoch'u ve checkpoint'i testten önce sabittir. Test sonucu aday seçimini değiştirmeyecektir.

## Neden iki checkpoint de değerlendirilecek?

Projenin araştırma sorusu yalnız en iyi modeli bulmak değil, eşit bütçeli GA ve random search
karşılaştırmasıdır. İki aday da validation kullanılarak önceden seçildiği için aynı test yollarında
tek seferlik değerlendirme, yönteme bağlı farkın paired olarak ölçülmesini sağlar.

## Dondurulmuş girdiler

- Practical split SHA-256: `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`.
- Confirmation summary SHA-256: `37c5d8a2f353642403a0d5469122a01c2c82fb915fd60b5efb0f7bd7744d91e1`.
- Confirmation run contract SHA-256: `c0d11243ac4ef50c305ea03b41fc13999f030039da5a068081ec872fd1aa0017`.
- Confirmation config SHA-256: `7f0886527f24365373842081561a0a4b6318f16fb59d0272fff1266a9bbd34a9`.
- Label taxonomy SHA-256: `63820de352eb442cacf7f9701ac6006ff3c16a26642eae5042015941d8d8dd1b`.
- Test sayısı: 19.500; her 26 leaf için 750.

Checkpoint dosyaları hafif review ZIP'inde bulunmadığından, ilk Colab preflight aşaması Drive'daki
iki `best_model.pth` dosyasını açmadan önce hash'lerini hesaplayacak, iç sınıf sırasını ve eğitim
argümanlarını validation sonuçlarıyla doğrulayacak ve `checkpoint_freeze.json` yazacaktır. Bu
aşamada dataset görüntüsü veya test satırı açılmaz. Uzun test hücresi aynı hash'leri yeniden
doğrulamadan çalışamaz.

## Değerlendirme çıktıları

Her aday için:

- leaf accuracy, macro-F1 ve balanced accuracy;
- türetilmiş sekiz-parent metrikleri;
- 26-leaf ve sekiz-parent classification report/confusion matrix;
- 19.500 satırlık path, true/predicted leaf-parent ve probability CSV'si.

Ortak karşılaştırmada aynı test yolları üzerinden:

- ikisinin de doğru olduğu görüntüler;
- yalnız GA'nın doğru olduğu görüntüler;
- yalnız random adayının doğru olduğu görüntüler;
- ikisinin de yanlış olduğu görüntüler;
- leaf ve parent için exact McNemar p-değeri;
- GA eksi random accuracy, macro-F1 ve balanced-accuracy farkı.

## Sınırlamalar

Bu test split'i daha önce baseline ve ablation modellerinde kullanılmıştır. Bu nedenle sonuç
“untouched final test” veya dış doğrulama olarak anlatılmayacaktır; aynı shared benchmark üzerinde
ek karşılaştırmadır. Tek seed, residual bilinmeyen near-duplicate/patient-series riski ve
modality/source shortcut riski devam eder. Sonuç klinik tanı kanıtı değildir.

## Çalıştırma kapısı

Notebook önce repo/config/confirmation/checkpoint preflight yapacaktır. Test değerlendirme hücresi
varsayılan olarak kilitli kalır. Mehmet preflight çıktısını Codex ile inceleyip ayrıca onaylamadan
`AUTHORIZE_SHARED_TEST = True` yapılmayacaktır.

## Dondurulmuş uygulama hash'leri

- Evaluator: `d11e0f3cac7b0a056ce87a163f83e8aa043206408352623e83170dd854bb7419`.
- Config: `9ac37b7c8d400eb363fecfc4801d4df835ab4f629e3a371a0a5e67998dec934e`.
- Colab notebook: `a2627da18dca3ebd673bb49072fa14e84a30d6c28f3b795b8a468901d1632596`.
- Odak evaluator testi: `40c9eb176977e0b4690bcf7d5861e00545d8b18653eaca1e401a067768b502f9`.
- Notebook testi: `68111cf218e8d86357c9fbf2149eb0c7cb385d33219e307c25f58e7cd3ccfc4c`.

Odak testler 8/8, bütün proje paketi 245/245 geçmiştir. Notebook temizdir; hiçbir hücre
execution count veya çıktı taşımaz.
