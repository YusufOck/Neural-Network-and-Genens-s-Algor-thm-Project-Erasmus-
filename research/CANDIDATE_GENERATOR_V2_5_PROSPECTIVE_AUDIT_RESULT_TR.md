# Candidate Generator v2.5 — Prospective Audit Sonucu

Tarih: 2026-11-11
İşlem türü: Önceden tanımlanmış, score-blind prospective validation gate değerlendirmesi
Durum: **V2.5 reddedildi**

## 1. Ne yapıldı?

Görsel kararlar private seçim bilgisi açılmadan önce dondurulmuştu. Bu değerlendirmede:

- 256 dondurulmuş görsel karar private selection strata ile Pair ID üzerinden birleştirildi;
- kararların hiçbiri değiştirilmedi;
- private/public kimlik alanları, Pair ID kapsamı ve source-group path uyumu doğrulandı;
- önceden belirlenen dokuz kabul kapısı tek raporda hesaplandı;
- duplicate scorer, family construction, split ve classifier training çalıştırılmadı.

## 2. Kör inceleme açıldığında ne görüldü?

| Bölge | Çift | Aynı-kaynak | Bağımsız | Belirsiz |
|---|---:|---:|---:|---:|
| V2.5 candidate | 80 | 20 | 59 | 1 |
| V2.5 outside | 176 | 16 | 160 | 0 |
| **Toplam** | **256** | **36** | **219** | **1** |

Candidate içindeki 20 aynı-kaynak çifti yöntemin bulabildiği örneklerdir; bunlar bir recall
hatası değildir. Buna karşılık outside bölgesindeki 16 aynı-kaynak çifti aday üreticinin hiç
göndermediği örneklerdir. Önceden belirlenen gerekli sayı sıfır olduğu için V2.5 reddedildi.

Outside kaçakların source-group dağılımı:

| Source group | Outside aynı-kaynak |
|---|---:|
| `brain_ce_mri` | 6 |
| `kidney_ct` | 6 |
| `lymphoma_histology` | 4 |
| Diğer beş grup | 0 |

Stratum dağılımı:

- `outside_robust_boundary`: 14 aynı-kaynak;
- `outside_random`: 2 aynı-kaynak;
- `outside_phash_13_16`: 0 aynı-kaynak.

Tek belirsiz çift `V25N000227`, candidate bölgesindedir. Bu nedenle “outside uncertain = 0”
kapısı geçmiştir.

## 3. Kabul kapıları

| Kapı | Gözlenen | Gerekli | Sonuç |
|---|---:|---:|---|
| Exact referans | 1.393/1.393 | 1.393/1.393 | Geçti |
| Kontrollü pozitif | 1.040/1.040 | 1.040/1.040 | Geçti |
| Eski doğal regression | 23/23 | 23/23 | Geçti |
| Yeni outside aynı-kaynak | 16 | 0 | **Kaldı** |
| Yeni outside unresolved uncertain | 0 | 0 | Geçti |
| Cross-source-group çift | 0 | 0 | Geçti |
| Full candidate sayısı | 2.994.870 | 2.994.870 | Geçti |
| Candidate yükü | 23,037/görüntü | ≤25 | Geçti |
| V2.4 baseline yeniden üretimi | 2.491.553 | 2.491.553 | Geçti |

Yalnızca bir kapı kaldı fakat bu kapı temel recall kapısıdır. Bu nedenle sonuç “sekiz kapı
geçti” diye kabul edilemez.

## 4. Bütünlük ve tekrar üretilebilirlik

- Frozen visual CSV SHA-256:
  `711bfd27668ca1f64eba8627d0973cf27561b577d2d72871cbd16b89e7b3a173`
- Private selection SHA-256:
  `b2b5da720ef256bc0c04a2aa788c4ff96de6ae372b1efd14997bbf45e9c4c462`
- Unblinded pair CSV SHA-256:
  `28a387c57e8b96c85f51b375c7c105dacb187b07e82bfba446d8d481e507958c`
- Unblinded JSON report SHA-256:
  `14d9df6ff3f03da9d9555324625e9934df274f568a736bfa669b0cb321cfbb01`
- Proje testleri: **137/137 başarılı**.

## 5. Bilimsel yorum

Bu sonuç V2.5 kuralının sampled challenge set üzerinde güvenli bir leakage-aware split için
yeterli olmadığını gösterir. Görsel kararlar AI-destekli, uzman-olmayan leakage-audit
etiketleridir; klinik ground truth değildir. Ayrıca örneklem bütün olası görüntü çiftlerini
ölçmez. Buna rağmen, kararlar açılmadan önce “outside pozitif = 0” kapısı dondurulduğu için 16
outside pozitif kuralı reddetmeye yeterlidir.

Sonucu gördükten sonra bu 16 kararı stratum bilgisine göre değiştirmek veya aynı audit üzerinde
eşiği yükseltip geçti saymak yasaktır.

## 6. En kısa savunulabilir sonraki adım

Bu 16 çift artık development kanıtıdır. Önce yalnız bu çiftlerin dört descriptor kanalındaki
full-index yönlü rank değerleri ölçülmelidir. Bu dar tanı:

1. brain MRI, kidney CT ve lymphoma için gereken minimum K değerlerini belirler;
2. yeni candidate yükünü tam olarak projekte eder;
3. günler sürebilecek yanlış scorer koşusunu önler;
4. sonuç görülmeden dondurulacak V2.6 kuralı ve yeni path-disjoint audit tasarımına girdi olur.

V2.5 altında candidate scoring, family construction, train/validation/test split veya model
eğitimi başlatılmayacaktır.
