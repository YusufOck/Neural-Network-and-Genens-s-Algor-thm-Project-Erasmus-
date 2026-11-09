# Candidate Generator v2.5 — Prospective Selection Sonucu

Tarih: 2026-11-09

İşlem türü: Full-index score-blind pair selection

Durum: Selection tamamlandı ve artifact'lar donduruldu; görsel audit henüz başlamadı

## 1. Çalıştırma ortamı

- Colab commit: `a59df105058f5dc9d14b9ae94b1604380c37919f`
- Cihaz: Tesla T4 / CUDA
- Frozen config ve builder hash kontrolü: geçti
- V2.3 kayıtlı artifact: 1.107
- Bu selection tarafından kullanılan ve hash ile doğrulanan artifact: 910
- Return code: 0

910 artifact hesabı:

- 1 full image index
- 26 descriptor artifact'ı
- 26 pHash-generation artifact'ı
- 32 robust top-15 artifact'ı
- pHash açık altı gruba ait 825 pHash-retrieval block artifact'ı

MRI ve CT pHash kanalı v2.5'te kapalı olduğu için bu iki grubun pHash-retrieval block'ları
selection girdisi olarak kullanılmadı. Bu eksiklik değil, frozen policy'nin sonucudur.

## 2. Paket ve artifact bütünlüğü

Colab paketi:

- ZIP SHA-256: `d5259e1a48cdb94a904540a5abc9fce1060e4bdc6871683679a465e8cd78d588`

Git'e alınan artifact'lar:

| Dosya | SHA-256 |
|---|---|
| `selection_report.json` | `0d07aa5a91a7cd0c21f6bd35f1e0899016247dbf7768eb0c3557041a809eb33b` |
| `selection_private.csv` | `b2b5da720ef256bc0c04a2aa788c4ff96de6ae372b1efd14997bbf45e9c4c462` |
| `review_manifest.csv` | `b551ef6bea86ee62c9fc831a48dd0b662a99344f25da7ec9ed550a2a27103bfb` |

`selection_private.csv` satırları denetim sırasında parse edilmedi veya gösterilmedi; yalnız ham
dosya SHA-256 değeri raporla karşılaştırıldı. `review_manifest.csv` score/stratum alanları
taşımadığı için yapısal ve boş-karar kontrollerinde okunabildi.

Bağımsız ZIP, hash, şema, aritmetik, exclusion ve yerel dosya kontrolü: **39/39 geçti**.

## 3. Candidate policy yeniden üretimi

| Grup | K | V2.4 union | V2.5 union |
|---|---:|---:|---:|
| ALL blood smear | 15 | 422.718 | 422.718 |
| Brain CE-MRI | 15 | 272.808 | 272.808 |
| Breast histology | 15 | 277.878 | 277.878 |
| Cervix Pap smear | **40** | 378.508 | **881.825** |
| Kidney CT | 15 | 115.349 | 115.349 |
| Lung/colon histology | 15 | 461.152 | 461.152 |
| Lymphoma histology | 15 | 339.509 | 339.509 |
| Oral histology | 15 | 223.631 | 223.631 |

Yalnız cervix grubu değişti. Full sonuç:

- V2.4: 2.491.553
- V2.5: 2.994.870
- Artış: 503.317
- V2.5 yük: 23,0374615 çift/görüntü
- 25 çift/görüntü kapasite kapısı: geçti
- Cross-group üretim: yok; bütün grup union toplamı full toplamla birebir eşleşti

## 4. Exclusion ve kör seçim

- Full index: 130.000 görüntü
- Frozen exclusion: 22.922 görüntü yolu
- Eligible: 107.078 görüntü
- Seçilen çift: 256/256
- Kullanılan benzersiz görüntü: 512
- Bir görüntünün iki çiftte kullanımı: 0
- Exclusion ile kesişen seçili yol: 0
- Yerelde bulunamayan seçili yol: 0
- Review Pair ID: 256/256 benzersiz, `V25N` önekli
- Review karar alanı: 256/256 boş
- Review CSV'ye sızan private score/stratum alanı: 0

## 5. Stratum kapsamı

| Stratum | Hedef | Gerçek | Eksik |
|---|---:|---:|---:|
| `outside_robust_boundary` | 64 | 64 | 0 |
| `outside_phash_13_16` | 48 | 48 | 0 |
| `outside_random` | 64 | 64 | 0 |
| `candidate_new_cervix_rank16_30` | 16 | 16 | 0 |
| `candidate_robust_edge` | 64 | 64 | 0 |
| **Toplam** | **256** | **256** | **0** |

Source-group dağılımı planla birebir eşleşti: altı pHash-açık grup 32'şer, MRI ve CT 24'er,
cervix ek yeni-band örnekleri nedeniyle 48 çift aldı.

## 6. Bu sonuç neyi kanıtlar?

Bu sonuç şunları kanıtlar:

1. Frozen v2.5 selection kodu planlanan full candidate yükünü yeniden üretti.
2. Yeni prospective audit, önceki geliştirme/audit/exact görüntülerinden ayrık 256 çift hazırladı.
3. Review paketi kördür ve henüz hiçbir karar içermez.
4. Teknik selection kapıları eksiksiz geçti.

Bu sonuç şunları **kanıtlamaz**:

- outside katmanlarında aynı-kaynak çift bulunmadığını;
- v2.5'in doğal near-duplicate recall kapısını geçtiğini;
- seçilen çiftlerin duplicate veya bağımsız olduğunu;
- family/split/model eğitimine geçilebileceğini.

## 7. Sonraki kontrollü adım

Artifact'lar GitHub'da dondurulduktan sonra yalnız `review_manifest.csv` kullanılarak score-blind
temas sayfaları render edilecektir. Renderer private CSV'yi okumamalı ve kartlarda yalnız Pair ID,
source group ve iki görüntüyü göstermelidir. Görsel kararlar tamamlanıp ayrıca dondurulmadan private
stratum/skorlar açılmayacaktır.

