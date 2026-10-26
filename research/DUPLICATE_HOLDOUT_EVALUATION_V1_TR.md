# Duplicate rule v1 holdout değerlendirmesi

Tarih: 2026-10-26

Durum: **671/671 holdout çifti, development'ta dondurulmuş kural değiştirilmeden bir kez
değerlendirildi; istemci kullanım limiti kesintisinden sonra kaydedilmiş çıktılar bağımsız
doğrulayıcıyla eksiksiz bulundu.**

Bu sonuç kanser sınıflandırma performansı değildir. Amaç, aynı kaynak görüntünün parlaklık,
döndürme, kaydırma, kırpma veya benzeri dönüşümlerini bulacak duplicate karar kuralını sınamaktır.

## 1. Kör inceleme neden önce yapıldı?

Holdout, yöntem için daha önce görülmemiş sınavdır. Aşağıdaki sıra korundu:

1. Duplicate rule v1 yalnız development kanıtıyla donduruldu.
2. Kural dosyasının SHA-256 değeri kaydedildi.
3. Holdout'taki 139 doğal aday için pHash, ORB, SSIM, NMI ve korelasyon değerlerini göstermeyen
   kör temas sayfaları üretildi.
4. Görüntüler yalnız nadir yerel yapıların örtüşmesine göre incelendi.
5. 139 karar Git ile izlenen `research/holdout_visual_review_v1.csv` dosyasına kaydedildi.
6. Ancak bundan sonra aynı scorer ve dondurulmuş kural holdout üzerinde çalıştırıldı.

Kör görsel referans dağılımı:

| Görsel karar | Sayı |
|---|---:|
| `same_source_candidate` | 15 |
| `independent_lookalike` | 124 |
| `uncertain` | 0 |
| Toplam | 139 |

Bu kararlar Codex tarafından, Mehmet'in açık yetkilendirmesiyle yapılan AI-assisted görsel
incelemedir. Bağımsız bir radyolog/patolog gold standardı değildir.

## 2. Çalıştırılan komutlar

Kör paket:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\prepare_holdout_blind_review.py
```

Tek seferlik holdout değerlendirmesi:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\evaluate_duplicate_holdout.py
```

İstemci kesintisinden sonra görüntüleri yeniden açmadan ve holdout'u yeniden skorlamadan yapılan
doğrulama:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\verify_duplicate_holdout.py
```

Yerel çıktılar:

- `artifacts/audit/duplicate_calibration_v1/holdout_blind_review_v1/`
- `artifacts/audit/duplicate_holdout_evaluation_v1/pair_scores_and_decisions.csv`
- `artifacts/audit/duplicate_holdout_evaluation_v1/report.json`
- `artifacts/audit/duplicate_holdout_evaluation_v1/verification.json`

`artifacts/` Git'e gönderilmez; bu belge doğrulanmış özetin Git ile izlenen kopyasıdır.

## 3. Bütünlük zinciri

Değerlendirme başlamadan önce iki kritik koruma geçti:

- frozen rule SHA-256:
  `5ffa80daa6e6ab0eb5f3ac179073de2f0f018325310310b33adc4c137710b645`
- development scorer ve holdout scorer SHA-256:
  `ac9f0a1538a24615bea6fac530806c1536aefb3ad0916fd3f2895d8ca4af513f`

Kesinti sonrası doğrulayıcı şu kontrolleri yaptı:

- 671 skor satırı ve 671 benzersiz Pair ID;
- beklenen 671 holdout Pair ID'sine karşı eksik 0, fazla 0;
- development satırı 0;
- boş referans veya boş otomatik karar 0;
- 671/671 kararın kaydedilmiş metriklerden frozen rule ile yeniden üretilebilmesi;
- frozen rule, scorer, görsel karar CSV'si ve ana pairs CSV hash'lerinin raporla eşleşmesi;
- genel, modalite ve pair-kind metriklerinin skor CSV'sinden yeniden üretilebilmesi;
- Python 3.10 test sonucu: 40/40 başarılı.

## 4. Holdout sonucu

Referans dağılımı 547 aynı-kaynak ve 124 bağımsız çifttir. Aynı-kaynakların 532'si kontrollü
sentetik/exact pozitif, 15'i kör görsel incelemede bulunan doğal adaydır.

| Referans | `same_source` kararı | `independent` kararı | `uncertain` |
|---|---:|---:|---:|
| Aynı kaynak (547) | **547** | 0 | 0 |
| Bağımsız (124) | 0 | **120** | 4 |

Türetilen ölçüler:

| Ölçü | Değer | Basit anlamı |
|---|---:|---|
| Kapsama (kesin karar verilen) | %99,40 | 671 çiftin 667'sinde kesin karar verildi |
| `uncertain` dahil genel doğruluk | %99,40 | Belirsizleri doğru saymadan 667/671 |
| Yalnız kesin kararlarda doğruluk | %100 | Kesin karar verilen 667 çiftte hata yok |
| Aynı-kaynak recall | %100 | 547 aynı-kaynak çiftin tamamı bulundu |
| Aynı-kaynak precision | %100 | Aynı-kaynak denilen hiçbir çift bağımsız değildi |
| Bağımsız recall | %96,77 | 124 bağımsız çiftin 120'si kesin negatif, 4'ü belirsiz |

Yalnız 139 kör görsel aday alt kümesinde kapsama ve belirsizleri yanlış sayan doğruluk
135/139 = **%97,12**'dir. Kesin karar verilen 135 doğal adayın tamamı görsel referansla
uyumludur. Bu alt küme, 520 kontrollü sentetik pozitifin genel sonucu şişirmesini ayırarak
yöntemin doğal zor adaylardaki davranışını daha açık gösterir.

## 5. Dört `uncertain` çift

Kural dört bağımsız görsel referansı zorla negatif yapmak yerine güvenli bölgede bıraktı:

| Pair ID | Modalite | Neden sınırda? |
|---|---|---|
| `C001213` | MRI | ORB inlier 27; bağımsız eşiği 20'nin üstünde, güçlü geometri koşulları yetersiz |
| `C001269` | MRI | pHash 8 ve ORB inlier 29; fakat oran/kapsama güçlü aynı-kaynak için yetersiz |
| `C001290` | CT | ORB inlier 29, SSIM 0,687; kesin iki bölgenin arasında |
| `C001291` | CT | ORB inlier 38, SSIM 0,723, korelasyon 0,943; görüntüler benzer fakat güçlü aynı-kaynak kanıtı yok |

Bu dört çift yanlış pozitif değildir. `uncertain`, “bilmiyorum; insan incelemesi gerekli” çıkışıdır.
Özellikle CT/MRI'da benzer veya komşu kesitler aynı 2D görüntü dönüşümü olmadan da çok benzer
görünebilir. Yerel veri hasta/seri kimliği içermediği için bu çalışma hasta düzeyi bağımsızlık
kanıtlamaz.

## 6. Modalite özeti

| Modalite | Çift | Kesin karar | `uncertain` | Kesin kararlarda hata |
|---|---:|---:|---:|---:|
| Blood smear | 102 | 102 | 0 | 0 |
| CT | 51 | 49 | 2 | 0 |
| Histopathology | 313 | 313 | 0 | 0 |
| MRI | 75 | 73 | 2 | 0 |
| Pap smear | 130 | 130 | 0 | 0 |

Bu tablo CT ve MRI için belirsiz bölgenin gerçekten kullanıldığını, güvenli fallback'in gereksiz
olmadığını gösterir. Bu aşamada eşikler değiştirilmemiştir.

## 7. Sonuç neyi kanıtlar, neyi kanıtlamaz?

Kanıtlar:

- Development'ta dondurulan açıklanabilir üç-çıkışlı kural, bu holdout çiftlerinde aynı-kaynak
  referansını kaçırmadı ve hiçbir bağımsız çifte yanlış `same_source` demedi.
- Tek pHash eşiği yerine ORB/RANSAC ve hizalanmış yapı ölçülerini birlikte kullanmak savunulabilir.
- `uncertain` çıkışı zor MRI/CT çiftlerinde güvenli biçimde devreye girdi.

Kanıtlamaz:

- Bu oranlar kanser sınıflandırma accuracy/macro-F1 sonucu değildir.
- 130.000 görüntünün tamamında doğal duplicate yaygınlığını ölçmez.
- Hasta/slide/seri ayrımını kanıtlamaz.
- AI-assisted görsel referans bağımsız klinik uzman değerlendirmesi değildir.
- Henüz augmentation family, 70/15/15 split veya model eğitimi üretmez.

## 8. Sonraki kontrollü adım

Rule v1 holdout doğrulaması tamamlanmıştır. Sonraki büyük adım, aynı aday üretme ve doğrulama
protokolünü 26 alt sınıfın tamamındaki 130.000 görüntüye ölçeklemek, uyumlu kaynak gruplarında
sınıflar arası adayları da aramak ve doğrulanmış exact/yakın kopyalardan parçalanmayan aileler
oluşturmaktır. Bu pahalı adım ayrı bir plan/kapasite kontrolünden sonra çalıştırılmalıdır; henüz
başlatılmamıştır.
