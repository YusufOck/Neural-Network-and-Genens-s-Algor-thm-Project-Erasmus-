# Candidate Generator v2.5 — Prospective Doğal Doğrulama Planı

Tarih: 2026-11-09  
İşlem türü: Deney tasarımı / score-blind audit planı  
Durum: Planlandı; config, seçim kodu ve çalışma henüz başlamadı

## 1. Neden doğrudan K=40'a geçmiyoruz?

Full-index rank-crowding tanısı `N000123` için minimum robust K değerini 34 ölçtü. Önceden
tanımlanmış grid içinde bu çifti ilk geri getiren nokta K=40'tır ve 2.994.870 aday = 23,037
çift/görüntü ile 25 sınırını geçmez.

Ancak bu sonucu gördükten sonra K=40 seçmek post-hoc bir tasarım kararıdır. Aynı `N000123`
çiftini hem kuralı seçmek hem de kuralı doğrulamak için kullanırsak gerçek bir sınav yapmış
olmayız. Bu yüzden:

- `N000123` ve önceki 23 doğal pozitif artık yalnız **development/regression kanıtıdır**;
- v2.5'in doğal risk kapısı, daha önce kullanılmamış görüntülerden yeni ve score-blind bir audit
  ile değerlendirilecektir;
- yeni audit görülmeden kural tekrar değiştirilmeyecektir.

## 2. Onay bekleyen v2.5 hipotezi

Bu plan onaylanırsa sonuç görülmeden config olarak dondurulacak öneri:

1. Dört-kanal robust ResNet18 retrieval, `cervix_pap_smear` grubunda top-40 olacaktır.
2. Diğer yedi source group robust top-15 olarak kalacaktır.
3. Flip-aware pHash≤12 şu altı grupta korunacaktır: ALL kan yayması, breast histology,
   cervix Pap smear, lung/colon histology, lymphoma histology ve oral histology.
4. pHash, `brain_ce_mri` ve `kidney_ct` gruplarında kapalı kalacaktır.
5. Bilinen exact SHA-256 çiftleri zorunlu olarak birleşime eklenecektir.
6. Source group dışı candidate üretmek yasaktır.

K=34 önerilmez: tek bilinen kaçağın ölçülen rank'ına tam oturur, önceden belirlenmiş yük grid'i
değildir ve exact yükü hesaplanmamıştır. K=40, önceden belirlenmiş grid'deki ilk recovery noktasıdır
ve altı rank'lık küçük bir tampon sağlar. Bu gerekçe yine de yeni audit zorunluluğunu kaldırmaz.

## 3. Değişmeyen beklenen kapasite

Rank-crowding tanısındaki exact counterfactual:

| Ölçü | Beklenen |
|---|---:|
| Full candidate | 2.994.870 |
| Çift/görüntü | 23,0375 |
| 25 sınırına kalan pay | 255.130 çift |
| Frozen scorer seri projeksiyonu | 28,25 gün |
| Sekiz worker ideal alt sınırı | 3,53 gün |

Son iki süre ölçülmüş paralel çalışma süresi değildir; `0,815 saniye/çift` eski scorer
projeksiyonundan hesaplanan kapasite göstergesidir. Audit geçse bile milyonlarca çifti skorlamak
için ayrıca çalıştırma/parallelization planı gerekir.

## 4. Yeni audit havuzu

Yeni seçim, full 130.000-image index ve mevcut frozen descriptor/pHash artifact'larından yapılır;
görüntüler seçim sırasında açılmaz. Aşağıdakiler tamamen dışlanır:

- önceki natural audit'in 6.604 görüntülük indeksindeki bütün yollar;
- v2.1/v2.2 kontrollü holdout ve scaled-pilot kaynak yolları;
- kalibrasyon/development/holdout çiftlerinde kullanılan kaynak yolları;
- bilinen exact SHA-256 duplicate grup üyeleri;
- `N000123`'ün iki yolu ve diğer 22 eski doğal pozitifin yolları;
- aynı yeni audit içinde daha önce seçilmiş bir görüntü.

Son madde nedeniyle her görüntü en fazla bir yeni çiftte yer alır. Her çift yalnız aynı source group
içinden seçilir; cross-leaf çiftlere source group uyumluysa izin verilir.

## 5. Önceden belirlenen score-blind örneklem

Seed: `20261109`. Pair ID biçimi: `V25N000001`.

Buradaki robust rank, dört descriptor kanalındaki iki yönün en küçük stable rank'ıdır; eşit cosine
durumunda mevcut frozen index sırası tie-break olarak kullanılır. Katman sınırları:

- `candidate_robust_edge`: cervix için rank 31–40, diğer gruplar için 6–15;
- `outside_robust_boundary`: cervix için rank 41–80, diğer gruplar için 16–30;
- `candidate_new_cervix_rank16_30`: yalnız rank 16–30;
- pHash-açık gruplardaki robust-only/new çiftlerde pHash mesafesi >12 olmalıdır;
- `outside_phash_13_16`: pHash 13–16 ve robust rank kabul sınırının dışında olmalıdır;
- `outside_random`: pHash-açık gruplarda robust rank `2K`'dan büyük ve pHash>16; MRI/CT'de
  yalnız robust rank `2K`'dan büyük olmalıdır.

Her pHash-açık source group için:

| Stratum | Hedef/grup | Amaç |
|---|---:|---|
| `candidate_robust_edge` | 8 | Robust listenin kabul sınırına yakın, pHash desteği olmayan çiftler |
| `outside_robust_boundary` | 8 | Kabul K'sının hemen dışındaki çiftler: cervix 41–80, diğerleri 16–30 |
| `outside_phash_13_16` | en çok 8 | pHash≤12 sınırının hemen dışı ve robust kuralın da dışında |
| `outside_random` | 8 | İki yöntemin uzağındaki taban kontrolü |

MRI ve CT için pHash kapalı olduğundan yapay bir pHash stratum'u üretilmez:

| Stratum | Hedef/grup |
|---|---:|
| `candidate_robust_edge` | 8 |
| `outside_robust_boundary` | 8 |
| `outside_random` | 8 |

Cervix grubuna ayrıca 16 adet `candidate_new_cervix_rank16_30` çifti eklenir. Cervix
`candidate_robust_edge` katmanı rank 31–40'tan seçilir. Böylece yeni eklenen 16–40 bandı yalnız
tek uçtan örneklenmez.

Planlanan üst sınır:

`6 pHash-açık grup × 32 + 2 pHash-kapalı grup × 24 + 16 cervix ek = 256 çift`

Bir stratum'da yeterli uygun çift yoksa başka stratum'dan tamamlanmaz; gerçek sayı raporlanır.

## 6. Körlük ve karar kuralları

1. Private seçim manifesti rank, cosine, pHash ve stratum bilgisini taşır.
2. Review manifesti ve temas sayfaları yalnız Pair ID, source group ve iki görüntüyü gösterir.
3. Seçim kodu/config, private manifest ve boş review manifesti görüntüler açılmadan önce hash ile
   dondurulup GitHub'a push edilir.
4. Kararlar `same_source_candidate`, `independent_lookalike` veya `uncertain` olur.
5. İlk görsel kararlar AI-destekli uzman-olmayan leakage audit etiketleridir; klinik gold standard
   değildir.
6. Private skorlar ancak bütün görsel kararlar dondurulduktan sonra açılır.
7. `uncertain` çiftler ayrı adjudication kuyruğuna alınır. Dış katmanda çözülmemiş `uncertain`
   varken audit geçmiş sayılmaz.

## 7. Önceden belirlenen kabul kapıları

| Kapı | Gerekli sonuç |
|---|---:|
| Exact SHA-256 referansları | 1.393/1.393 |
| Kontrollü sentetik pozitifler | 1.040/1.040 |
| Eski doğal regression referansları | 23/23; bağımsız test sayılmaz |
| Yeni `outside_*` katmanlarında kesin aynı-kaynak | 0 |
| Yeni `outside_*` katmanlarında çözülmemiş uncertain | 0 |
| Source group dışı candidate | 0 |
| Full candidate sayısı | tam 2.994.870 |
| Candidate yükü | en fazla 25 çift/görüntü |
| Frozen K=15 baseline yeniden üretimi | 2.491.553 |

Candidate katmanında bulunan aynı-kaynak çiftler olumlu recall kanıtıdır. Bağımsız benzerlerin çok
olması tek başına candidate generator hatası değildir; precision sonraki duplicate scorer'ın
işidir. Buna karşılık yeni `outside_*` katmanında bir kesin aynı-kaynak çift bulunursa v2.5
reddedilir.

Yeni outside örneklemde pozitif bulunmaması bütün olası çiftlerde %100 recall kanıtlamaz. Sonuç
yalnız örneklenmiş challenge set için risk azalması olarak raporlanacaktır.

## 8. Tek kullanımlık holdout disiplini

- Yeni audit yalnız bir kez açılır.
- Bir outside pozitif bulunursa aynı review setine bakarak K veya pHash eşiği tekrar ayarlanmaz.
- Yeni kural gerekirse v2.6 adıyla, yeni path-disjoint audit ile tasarlanır.
- N000123 v2.5'in 23/23 regression kapısında bulunmalıdır fakat prospective başarı sayılmaz.

## 9. Uygulama adımları ve durma noktaları

1. **Şu an:** Bu plan Mehmet ile incelenir; henüz kod yoktur.
2. Ayrı onaydan sonra v2.5 config, selection config ve testler yazılır; hash'ler dondurulur.
3. Mevcut Drive artifact'larından full membership/yük ve score-blind seçim manifesti üretilir.
4. Manifest bütünlüğü doğrulanıp, görüntüler açılmadan commit/push yapılır.
5. Dataset görüntüleri kullanılarak temas sayfaları render edilir.
6. 256'ya kadar çift score-blind incelenir ve karar dosyası dondurulur.
7. Private katmanlar açılır; bütün kapılar tek raporda değerlendirilir.
8. Ancak audit geçerse duplicate scoring çalıştırma planı hazırlanır.

## 10. Bu planın yetki vermediği işler

- v2.5 config veya seçim kodunu henüz yazmak;
- full candidate dosyalarını üretmek;
- duplicate scorer çalıştırmak;
- augmentation family oluşturmak;
- train/validation/test split yapmak;
- classifier eğitmek;
- genetik algoritma veya Grad-CAM başlatmak.

