# Natural near-duplicate doğrulama planı v1

Tarih: 2026-10-29

Durum: **Planlandı; çiftler seçilmedi ve görüntüler açılmadı.** Bu bir veri
sızıntısı audit planıdır. Model eğitimi, family oluşturma veya train/validation/
test split üretme adımı değildir.

## 1. Neden bu adım gerekli?

Candidate Generator v2.1 ve v2.2, birbirinden yol olarak ayrık iki kontrollü
holdout'ta 1.040 pozitifin 1.039'unu buldu. Her iki yöntem de aday yükü sınırını
geçti, fakat önceden ilan edilen tam-recall kapısını bir çiftle kaçırdı.

Tek bir kaçak, sınıflandırma metriğinde küçük görünebilir. Burada ise kaçan çift
aynı kaynak görüntünün iki akrabasıysa ve ileride farklı split'lere düşerse data
leakage oluşturabilir. Bu nedenle sonuçlar kötü sayılmaz, fakat leakage-aware
split için henüz yeterli güvenlik kanıtı sayılmaz.

V2.2 sonucu görüldükten sonra `robust top-15 union pHash<=12`, aynı kontrollü
holdout'ta 1.040/1.040 ve 22,501 çift/görüntü sağladı. Bu yalnızca v2.3
hipotezidir. Üçüncü kez aynı on sentetik dönüşüme göre ayar yapmak, yöntemi gerçek
dataset yerine dönüşüm üreticisine uydurma riski taşır.

## 2. Bu audit hangi soruları cevaplayacak?

1. V2.3 hipotezinin aday listesine giren gerçek dataset çiftleri içinde doğal
   aynı-kaynak akrabalar var mı?
2. Kuralın hemen dışında kalan zor çiftlerde aynı-kaynak akrabalar kaçıyor mu?
3. Farklı kaynak gruplarında robust ve pHash kanalları nasıl davranıyor?
4. Dondurulmuş duplicate karar kuralı ile kör görsel kararlar nerede uyuşuyor veya
   ayrışıyor?

Bu audit bütün 3.093.466 olası dataset-dataset çiftini insan gözüyle incelemez ve
doğal recall'ı matematiksel olarak %100 kanıtlayamaz. Ama yalnız sentetik
dönüşümlere dayanmaktan daha güçlü, izlenebilir bir risk kontrolü sağlar.

## 3. Kullanılacak görüntü havuzu

V2.2 holdout indeksindeki yalnız gerçek dataset dosyaları kullanılacak:

- 6.604 dataset görüntüsü;
- 26/26 leaf ve 8/8 source group;
- scaled pilot ve v2.1 holdout'ta kullanılan dataset yollarıyla kesişim 0;
- bilinen SHA-256 exact-duplicate grup üyeleri dışlanmış;
- üretilmiş 1.040 kontrollü dönüşüm doğal audit havuzuna alınmayacak.

Bu 6.604 görüntü için source-group içindeki olası çift sayısı 3.093.466'dır.
V2.3 hipotezi altında ölçülen, henüz görsel olarak etiketlenmemiş havuz:

| Kanal | Dataset-dataset çift |
|---|---:|
| Robust top-15 | 132.117 |
| pHash<=12 | 6.079 |
| İki kanalın kesişimi | 1.548 |
| Birleşim | 136.648 |
| Birleşim yükü | 20,692 çift/dataset görüntüsü |

Bu sayımlar yeniden eğitim veya yeni görüntü etiketi kullanmaz; kaydedilmiş
descriptor ve hash çıktılarından hesaplanmıştır.

## 4. Önceden belirlenecek kör örneklem

Seed `20261030` ile, görüntüler açılmadan ve karar verilmeden önce sabit bir
manifest üretilecek. Pair ID'ler `N000001` biçiminde olacak. Her seçim yalnız
aynı uyumlu source group içinde yapılacak.

Her sekiz source group için:

| Stratum | Hedef çift/grup | Amaç |
|---|---:|---|
| `candidate_robust_only` | 12 | ResNet kısa listesinde olup pHash desteği olmayan doğal adaylar |
| `candidate_phash_supported` | en çok 8 | pHash-only ve iki-kanal adayları; mevcut sayı azsa tamamı |
| `outside_robust_rank16_30` | 8 | Top-15 dışında ama robust sıralamada hemen yakın çiftler |
| `outside_phash_distance14_16` | 8 | pHash<=12 sınırının hemen dışındaki çiftler |
| `outside_random` | 8 | İki yöntemin de uzağındaki taban kontrolü |

Beklenen üst sınır 345 çifttir. `breast_histology` grubunda yalnız bir
pHash<=12 dataset-dataset çifti bulunduğu için o stratum yapay olarak sekize
tamamlanmayacak. Eksik stratumlar başka bir kanaldan çift eklenerek gizlenmeyecek;
gerçek sayı raporlanacak.

## 5. Körlük ve inceleme kuralları

- Seçim manifestinde stratum ve skorlar saklanacak.
- İnceleme temas sayfalarında yalnız Pair ID, source group ve iki görüntü olacak;
  pHash uzaklığı, robust rank/benzerlik ve kanal adı gösterilmeyecek.
- Bütün karar alanları başlangıçta boş olacak.
- Kararlar `same_source_candidate`, `independent_lookalike` veya `uncertain`
  olacak.
- Görsel kararlar AI-destekli olacaktır; klinik uzman gold standard olarak
  sunulmayacaktır.
- Görsel karar manifesti dondurulduktan sonra frozen duplicate rule v1 skorları
  karşılaştırılacak. Uyuşmazlıklar otomatik olarak kesin family bağına
  dönüştürülmeyecek; `uncertain` inceleme kuyruğuna girecek.

## 6. Önceden belirlenen yorumlama kapısı

Bu audit iki farklı sonucu ayıracak:

1. **Candidate tarafı:** `candidate_*` strata içindeki aynı-kaynak çiftleri,
   yöntemin doğal akrabaları bulabildiğine dair kanıttır. Candidate precision'ın
   düşük olması tek başına başarısızlık değildir; aday üreticinin görevi yüksek
   recall ile kısa liste hazırlamaktır.
2. **Kural dışı taraf:** `outside_*` strata içinde kesin bir aynı-kaynak çift
   bulunursa v2.3 doğal risk kontrolünü geçmez. Kural genişletilmeden veya ek
   bağımsız kanal kurulmadan full scan başlatılmaz.

Kural dışı örneklemde hiç pozitif bulunmaması, bütün görünmeyen çiftlerde recall
%100 demek değildir. Rapor yalnız incelenen challenge set için sonuç verecek ve
bu sınırlamayı açıkça yazacaktır.

## 7. Aşamalar ve durma noktaları

1. Deterministik seçim kodu ve testleri yazılır.
2. Manifest oluşturulur; satır sayısı, benzersiz Pair ID, dosya varlığı ve input
   hash'leri doğrulanır.
3. Manifest ve seçim kodu Git'e commit/push edilerek seçim dondurulur.
4. Score-blind temas sayfaları oluşturulur.
5. Görsel inceleme tamamlanır ve karar dosyası ayrıca dondurulur.
6. Frozen duplicate rule v1 ile karşılaştırma yapılır.
7. Ancak sonuçtan sonra v2.3'ün yeni kontrollü holdout'a mı, daha geniş bir aday
   kanalına mı, yoksa belgelenmiş artık-riskle throughput pilotuna mı gideceğine
   karar verilir.

Şu anda yalnız plan tamamlanmıştır. 130.000 görüntülük tarama, family, split ve
model eğitimi başlamamıştır.
