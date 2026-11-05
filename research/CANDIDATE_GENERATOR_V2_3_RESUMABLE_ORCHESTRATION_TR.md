# Candidate Generator v2.3 resumable tam-retrieval altyapısı

Tarih: 2026-11-05

Durum: **Tam retrieval için hash-korumalı, parçalı ve resume edilebilir çalıştırıcı
hazırlandı; yerelde gerçek görüntülerle kesinti-devam smoke testi geçti. Tam
130.000 görüntülük retrieval başlatılmadı.** Duplicate-rule scoring, family,
train/validation/test split ve model eğitimi yoktur.

## 1. Neden yeni bir çalıştırıcı gerekti?

260 görüntülük önceki throughput shard tek çalışma içinde tamamlanıyordu. Tam veri
üzerinde descriptor üretimi, pHash ve source-group içi retrieval saatler sürebilir.
Colab bağlantısı kesildiğinde bütün işi sıfırdan yapmak bilimsel ve operasyonel olarak
uygun değildir.

Yeni çalıştırıcı her aşamayı küçük immutable artifact'lara ayırır. Her artifact
SHA-256 ile `stage_manifest.json` içine kaydedilir. Aynı komut tekrar çağrıldığında
dosya kullanılmadan önce hash'i doğrulanır; eşleşen aşama atlanır, değişmiş dosya
reddedilir.

## 2. Güvenlik modeli

- Varsayılan kapsam `smoke`tur.
- Tam kapsam yalnız `--scope full` ve ayrı
  `--confirm-full-run FROZEN_V2_3_130000` anahtarı birlikte verilirse açılır.
- `run_contract.json`; config, orchestrator, runtime, yedi frozen dependency,
  ResNet18 checkpoint, exact-duplicate manifest, veri envanteri, ortam sürümleri ve
  bütün önemli argümanları kilitler.
- Resume sırasında mevcut contract ile tek alan bile farklıysa çalışma durur.
- Kayıtlı artifact eksikse veya SHA-256 değişmişse çalışma durur.
- Kayıtsız mevcut final/stage dosyasının üzerine yazılmaz.
- Aday yükü 25 çift/görüntü sınırını aşarsa finalleştirme durur.
- Source-group dışı aday sayısı sıfır olmalıdır.
- Full kapsamda bilinen 1.393 byte-identical çiftin tamamı candidate union içinde
  bulunmalıdır.

## 3. Parçalanan aşamalar

1. Deterministik `image_index.csv`;
2. 26 leaf için ayrı pHash NPZ dosyası;
3. 26 leaf için ayrı dört-kanallı `(N, 512)` descriptor NPZ dosyası;
4. sekiz source group × dört descriptor kanalı için robust top-15 parçaları;
5. her source group için bloklu flip-aware pHash `<=12` parçaları;
6. tekrarsız robust/pHash birleşimi;
7. `candidate_pairs.npz`, `candidate_pairs.csv`, `run_report.json` ve
   `hash_manifest.json`.

Bu yapı sayesinde örneğin 17. leaf descriptor aşamasında kesinti olursa doğrulanmış
ilk 16 leaf yeniden hesaplanmaz.

## 4. Frozen yöntemle eşdeğerlik

Yeni runtime yöntemin parametresini değiştirmez:

- robust kanal hâlâ dört descriptor kanalının yönlü top-15 birleşimidir;
- cosine similarity aynı float32 descriptorlar üzerinde exact hesaplanır;
- hızlı top-k seçimi, eşit skorlar dâhil eski stable descending sort ile aynı indeks
  kümesini seçer;
- pHash kanalı orijinal/yatay-flip dört kombinasyonunun minimum exact Hamming
  uzaklığını kullanır;
- radius inclusive `12` ve yalnız uyumlu source-group içidir.

Birim testinde yeni robust seçimi eski frozen blockwise retrieval ile aynı çiftleri,
yeni bloklu pHash ise brute-force flip-aware yöntemle aynı çiftleri verdi.

## 5. Yerel kesinti-devam smoke testi

Kapsam: 26 leaf × 1 görüntü = 26 görüntü.

Birinci çağrı:

- indeks ve 26 leaf pHash artifact'ı üretildi;
- toplam 27 artifact kaydedildi;
- çalışma `--stop-after phash` ile bilinçli durduruldu.

İkinci çağrı:

- aynı contract ve output klasörü kullanıldı;
- 27/27 önceki artifact hash ile doğrulandı;
- indeks ve 26 pHash dosyası yeniden hesaplanmadan atlandı;
- descriptor, robust retrieval, pHash retrieval ve finalleştirme tamamlandı.

Sonuç:

| Ölçüm | Değer |
|---|---:|
| Smoke görüntüsü | 26 |
| Leaf | 26 |
| Source group | 8 |
| Candidate union | 35 |
| Çift/görüntü | 1,346 |
| Source-group dışı çift | 0 |
| Bu smoke kapsamındaki exact çift | 0 |
| Kayıtlı toplam artifact | 97 |
| Descriptor süresi | 9,071 saniye |

Bu küçük örnek hız veya aday yoğunluğu açısından full-run tahmini değildir. Amacı
kesinti-devam ve bütünlük davranışını doğrulamaktır.

## 6. Kod ve contract kimlikleri

| Öğe | SHA-256 |
|---|---|
| Full retrieval orchestrator | `b7f6dd9e4df2cb2314481ea5d9356fcf8dd0bfccbdaae21f5dcba355f6d734ec` |
| Resumable runtime | `2d3173100b08bc2d1b41b5c24cd9e90700c619fff00dba9777c5d6a35916cbf0` |
| Yerel smoke run contract | `cca22252e2a109254875f78fcddaf4c25c0abccdd40299607f31d231f450d8d6` |
| Frozen v2.3 config | `daef2f4d1c4199846a64b549a5070f5c8cc3f322cc949fb55dde2700ef2e3776` |
| Normalize exact-duplicate manifest | `e8a27465513e948d2cf7a012a4d2eaebea2dd97d8f5c313299f81eb86780ee4a` |

Proje test paketi: **97/97 geçti**.

## 7. Colab notebook

`notebooks/candidate_generator_v2_3_full_retrieval_colab.ipynb` şu sırayı zorlar:

1. Google Drive bağlanır;
2. `MyDrive/NNGA_DATA/Multi Cancer.zip` arşivi en az 10 GiB boş alan kontrolünden
   sonra Colab'ın hızlı `/content/NNGA_DATA` diskine çıkarılır;
3. sekiz beklenen ana klasörü içeren dataset kökü iki olası ZIP yerleşiminden
   otomatik bulunur;
4. repo klonlanır ve commit gösterilir;
5. T4 GPU doğrulanır;
6. exact-duplicate manifesti datasetten üretilir veya mevcut manifest doğrulanır;
7. önce 52 görüntülük Colab smoke çalıştırılır;
8. `run_report.json` ve `hash_manifest.json` Codex ile incelenir;
9. ancak ayrı `AUTHORIZE_FULL_RUN=True` kararı verilirse full hücresi açılır.

ZIP başarıyla açılmadan tamamlanma işareti yazılmaz. Runtime tamamen kapanırsa geçici
dataset yeniden çıkarılır; Drive'daki output/checkpoint artifact'ları korunur.

Full çalışma kesilirse yeni output adı verilmez ve kod/argümanlar değiştirilmez;
aynı hücre yeniden çalıştırılır. Contract veya ortam farklıysa runner güvenli biçimde
resume'i reddeder.

## 8. Sıradaki kullanıcı adımı

Tam retrieval henüz başlatılmayacaktır. Mehmet 8,62 GB `Multi Cancer.zip` dosyasını
`MyDrive/NNGA_DATA` altına yükledi. Güncellenmiş notebook Colab'da T4 ile açılacak;
ZIP çıkarma ve ortam kontrolünden sonra yalnız 52 görüntülük smoke bölümüne kadar
çalıştırılacaktır. Smoke `run_report.json` ve `hash_manifest.json` birlikte
incelendikten sonra full-run kararı ayrıca verilecektir.
