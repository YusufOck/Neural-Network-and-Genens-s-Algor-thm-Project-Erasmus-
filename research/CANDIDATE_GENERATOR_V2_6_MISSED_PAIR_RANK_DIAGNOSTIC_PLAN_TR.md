# Candidate Generator v2.6 — 16 Kaçak İçin Rank Tanı Planı

Tarih: 2026-11-11
İşlem türü: Önceden tanımlanmış, salt-okunur full-index rank diagnostic
Durum: Kod/config/notebook donduruldu; Drive tanısı henüz çalıştırılmadı

## Amaç

V2.5 prospective audit'te candidate alanı dışında kalan 16 `same_source_candidate` çiftin dört
frozen ResNet18 descriptor kanalındaki exact yönlü rank değerlerini ölçmek. Bu çiftler üç source
group içindedir:

- `brain_ce_mri`: 6 çift;
- `kidney_ct`: 6 çift;
- `lymphoma_histology`: 4 çift.

Bu işlem V2.6 kuralı seçmez. Sonuçlar yalnız yeni hipotez tasarımında development evidence olarak
kullanılacaktır.

## Dondurulmuş K grid'i

`[15, 20, 25, 30, 40, 50, 75, 100]`

Her source group için:

1. 16 hedef çiftin her descriptor kanalındaki iki yönlü exact rank'ı hesaplanır;
2. dört kanalın en iyi yönlü rank'ı belirlenir;
3. grubun bütün hedeflerini kapsayan minimum gözlenen K raporlanır;
4. her grid K için robust union yeniden kurulur;
5. V2.5 candidate yükünden başlayarak yeni full candidate yükü hesaplanır;
6. frozen top-15 union mevcut artifact ile birebir eşleşmek zorundadır.

Exact target rank 100'den büyük çıkabilir; exact rank yine raporlanır. Ancak candidate-load grid'i
sonuç görüldükten sonra keyfî biçimde genişletilmeyecektir.

## Kullanılan mevcut kanıtlar

- 1.107 hash-kayıtlı V2.3 full retrieval artifact'ı;
- dondurulmuş V2.5 unblinded CSV ve JSON;
- V2.5 selection report;
- üç grubun frozen descriptor, robust top-15 ve gerektiğinde pHash artifact'ları.

Dataset ZIP'i veya kaynak görüntüler gerekli değildir. Görüntüler açılmaz.

## Bütünlük kuralları

- Config'teki 16 Pair ID, frozen `outside + same_source_candidate` kümesiyle tam eşleşmelidir.
- Pair path, source group, stratum, pHash ve audit rank alanları değiştirilmemelidir.
- Stage manifest tam 1.107 artifact ve beklenen run-contract hash'ine sahip olmalıdır.
- Her grup için recomputed robust top-15, frozen robust artifact union ile birebir eşleşmelidir.
- Rapor immutable V2.3 output klasörünün dışında yazılmalıdır.

## Dondurulmuş dosyalar

- Analyzer SHA-256:
  `774f61d4542c3efe7ec23f62f2fa085b3fcb786c33fc5ae1c546894d3569707b`
- Config SHA-256:
  `21c6e164c0d2b9f28bb654d9534622ec75d5b99f14b93912cde260d8477645f7`
- Colab notebook SHA-256:
  `d19eb4ba6acd850a6303846d3d7af98a6a01f62fe617ce034d0e2a9d375a0ebc`

## Colab çıktısı

Drive'da tek ana rapor oluşacaktır:

`MyDrive/NNGA_OUTPUTS/candidate_generator_v2_6_missed_pair_rank_diagnostic_v1.json`

Notebook ayrıca timestamp'li console log tutar. Yaklaşık çalışma beklentisi T4 üzerinde 10–30
dakikadır; bu ölçülmüş garanti değildir.

Yerel doğrulamada notebook JSON'u geçerli bulundu, dört code cell derlendi ve proje testlerinin
141/141'i geçti. Bunlar full-index rank sonucu değil, çalıştırma altyapısı doğrulamasıdır.

## Bu planın yetki vermediği işler

- V2.6 kuralı seçmek veya kabul etmek;
- frozen görsel kararları değiştirmek;
- duplicate scorer çalıştırmak;
- family oluşturmak;
- train/validation/test split üretmek;
- classifier veya genetik algoritma eğitmek.

Drive raporu tamamlandıktan sonra exact rank ve kapasite sonuçları Mehmet ile incelenecek, ancak
ondan sonra ayrı onayla V2.6 politika planı hazırlanacaktır.
