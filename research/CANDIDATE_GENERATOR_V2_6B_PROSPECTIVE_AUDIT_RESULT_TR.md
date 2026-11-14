# Candidate Generator v2.6b — Prospective Audit Sonucu

Tarih: 2026-11-14

İşlem türü: Önceden tanımlanmış, score-blind, path-disjoint same-2D prospective audit

Durum: **V2.6b reddedildi**

## Ne yapıldı?

160 görsel karar private candidate/outside bilgisi açılmadan önce dondurulmuştu. Mehmet'in
ayrı unblinding onayından sonra:

- private selection, public manifest ve frozen kararlar Pair ID üzerinden birebir birleştirildi;
- 160/160 Pair ID kapsamı, kimlik alanları, same-leaf kapsamı ve source-group path uyumu
  doğrulandı;
- kararların hiçbiri değiştirilmedi;
- dokuz önceden tanımlı kabul kapısı hesaplandı;
- duplicate scorer, family construction, split ve model eğitimi çalıştırılmadı.

## Unblinding sonucu

| Bölge | Çift | Aynı-2B | Bağımsız | Belirsiz |
|---|---:|---:|---:|---:|
| V2.6b candidate | 32 | 9 | 23 | 0 |
| V2.6b outside | 128 | 15 | 112 | 1 |
| **Toplam** | **160** | **24** | **135** | **1** |

Candidate içindeki dokuz aynı-2B çift retrieval yönteminin bulabildiği örneklerdir. Outside
bölgesindeki 15 aynı-2B çift ise kuralın kaçırdığı örneklerdir. Önceden gerekli outside pozitif
sayısı sıfırdı; bu nedenle tek bir kaçağın bile kuralı reddetmesi gerekir. Ayrıca outside
bölgesindeki `V26P000007` belirsiz kaldığı için sıfır unresolved-uncertain kapısı da geçmedi.

Outside aynı-2B çiftlerin kaynak dağılımı:

| Source group | Outside aynı-2B |
|---|---:|
| `kidney_ct` | 5 |
| `lymphoma_histology` | 10 |

Stratum dağılımı:

- `outside_selected_boundary`: 12 aynı-2B, 52 bağımsız, 0 belirsiz;
- `outside_alternate_channel`: 3 aynı-2B, 28 bağımsız, 1 belirsiz;
- `outside_random_same_leaf`: 0 aynı-2B, 32 bağımsız, 0 belirsiz;
- `candidate_new_extension`: 9 aynı-2B, 23 bağımsız, 0 belirsiz.

## Kabul kapıları

| Kapı | Gözlenen | Gerekli | Sonuç |
|---|---:|---:|---|
| Exact referans | 1.393/1.393 | 1.393/1.393 | Geçti |
| Kontrollü pozitif | 1.040/1.040 | 1.040/1.040 | Geçti |
| Eski doğal regression | 23/23 | 23/23 | Geçti |
| Development same-2D | 4/4 | 4/4 | Geçti |
| Yeni outside aynı-2B | 15 | 0 | **Kaldı** |
| Outside unresolved uncertain | 1 | 0 | **Kaldı** |
| Cross-source-group çift | 0 | 0 | Geçti |
| Full candidate sayısı | 3.116.598 | 3.116.598 | Geçti |
| Candidate yükü | 23,974/görüntü | ≤25 | Geçti |

Yedi kapının geçmesi iki temel prospective kapının başarısızlığını telafi etmez. Sonucu
gördükten sonra eşik yükseltmek, karar değiştirmek veya aynı örneklem üzerinde yeni kuralı geçti
saymak bilimsel olarak geçerli değildir.

## Bütünlük ve tekrar üretilebilirlik

- Frozen visual CSV SHA-256:
  `2220fca301f0d281cab03f98847e865e02de1269b0881479ebae7813d7b4f36b`
- Private selection SHA-256:
  `549e12b57daac220fbfd631f2717c28f5b78370f42fb713afdb746e176a7f955`
- Unblinded pair CSV SHA-256:
  `e56de9ecd58d7254ea5f22eabb2cf35f1829581d61c869ea6e10bdc456ef891e`
- Unblinded JSON report SHA-256:
  `221a044e538d9dd71ca6f70f618a1892041f916918b8c44796c266d63fc8824c`
- Proje testleri: **173/173 başarılı**.

## Bilimsel sınır

Bu denetim `same_2d_augmentation_family_v1` hedefini test eder. Hasta veya seri kimliği
bulunmadığı için patient-level/series-level ayrımı kanıtlamaz. Görsel kararlar AI-destekli,
uzman-olmayan leakage-audit kararlarıdır; klinik ground truth değildir. Örneklenen 128 outside
çift bütün olası görüntü çiftlerini temsil etmez, fakat önceden dondurulan sıfır-kaçak kapısını
reddetmek için 15 pozitif yeterlidir.

## Sonraki karar noktası

V2.6b altında 3.116.598 aday skorlanmayacak, family oluşturulmayacak ve split/eğitim
başlatılmayacaktır. Teslim süresi nedeniyle sıradaki adım yeni bir sınırsız retrieval döngüsü
değil, mevcut kanıtları kullanarak savunulabilir ve uygulanabilir split stratejisini seçmektir.
Bu strateji ayrıca onaylanmadan uygulanmayacaktır.
