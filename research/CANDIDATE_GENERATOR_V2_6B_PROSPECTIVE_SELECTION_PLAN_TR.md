# Candidate Generator v2.6b Prospective Selection Planı

Tarih: 2026-11-13
Durum: config/builder/notebook hash-kilitli; full selection koşusu başlamadı

## Amaç

V2.6a'nın seçtiği `kidney_all_lymphoma_raw_multicrop` profilini, onu geliştirmek için
kullanılmamış yeni görüntüler üzerinde score-blind sınamak.

## Ayrıklık

- Önceki calibration/holdout/pilot/exact/natural kaynaklarından 22.922 yol dışlanır.
- V2.5 prospective audit'indeki 256 çiftin 512 benzersiz görüntüsü ayrıca dışlanır.
- İki küme kesişmez; toplam 23.434 yol dışlanır.
- Birleşik canonical path hash'i:
  `28a3c575e61a303548cadc9473edb35f1c23af1f8cf5cc159bf8f90d247254ba`.
- Kalan eligible görüntü: 106.566.

## Dondurulmuş 160-çift örneklemi

- 64 `outside_selected_boundary`: Kidney ve Lymphoma'dan 32'şer.
- 32 `outside_alternate_channel`: Lymphoma'da seçilmeyen descriptor kanalları.
- 32 `outside_random_same_leaf`: iki gruptan 16'şar.
- 32 `candidate_new_extension`: iki gruptan 16'şar.
- Toplam outside challenge: 128; candidate kontrol: 32.
- Her görüntü yalnız bir çiftte kullanılır; hedef eksikleri başka strata ile doldurulmaz.

## Körleme ve kapı

Public review manifest; stratum, candidate üyeliği, rank, kanal ve pHash skorlarını içermez.
Private manifest tüm bu alanları korur fakat görsel kararlar dondurulana kadar açılmaz.
Yeni 128 outside çift içinde doğrulanmış same-2D sayısı 0 ve çözümsüz uncertain sayısı 0
olmalıdır. Sıfır örneklenmiş kaçak bütün olası çiftlerde yüzde 100 recall kanıtı değildir.

## Yapılmayanlar

Selection sırasında görüntü açılmaz, duplicate scorer çalışmaz, family/split oluşturulmaz ve
classifier eğitilmez. Dataset ZIP'i seçim için gerekli değildir.
