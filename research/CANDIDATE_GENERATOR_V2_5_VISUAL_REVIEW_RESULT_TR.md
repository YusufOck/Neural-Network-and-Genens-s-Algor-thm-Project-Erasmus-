# Candidate Generator v2.5 — Score-Blind Görsel İnceleme Sonucu

Tarih: 2026-11-11

İşlem türü: Prospective score-blind doğal near-duplicate audit'i

Durum: İlk görsel kararlar tamamlandı ve donduruldu; private katman henüz açılmadı

## 1. İnceleme kapsamı

- 43/43 temas sayfası incelendi.
- 256/256 çift için karar kaydedildi.
- Sınırdaki çiftler 512×512 orijinal dosyalarda ikinci kez kontrol edildi.
- Görünen bilgi yalnız Pair ID, source group, L/R görüntüler ve public path/leaf alanlarıydı.
- `selection_private.csv` parse edilmedi veya gösterilmedi.

Bu kararlar, Mehmet'in açık yetkisiyle yapılan Codex AI-destekli uzman-olmayan leakage audit
etiketleridir. Klinik veya patolog gold standard değildir.

## 2. Dondurulmuş kararlar

| Görsel karar | Sayı |
|---|---:|
| `same_source_candidate` | 36 |
| `independent_lookalike` | 219 |
| `uncertain` | 1 |
| **Toplam** | **256** |

Tek belirsiz çift `V25N000227`'dir. İki hücrenin şekil ve yerleşimi benzer olsa da çevredeki kırmızı
kan hücresi işaretleri güvenilir bir birebir kaynak eşleşmesi göstermediği için zorla karar
verilmedi.

CT/MRI çiftlerinde yalnız aynı kesite benzeme değil, aynı acquisition/source ailesini düşündüren
tekrarlanabilir anatomi de leakage riski olarak `same_source_candidate` kapsamında tutuldu. Bu
yaklaşım, aynı hastaya veya aynı taramaya ait çok yakın kesitlerin ileride farklı split'lere
dağılmasını önlemek için korumacıdır.

## 3. Bütünlük ve testler

- Karar satırı: 256
- Benzersiz Pair ID: 256
- Public manifest ile Pair ID eşleşmesi: tam
- Boş karar: 0
- Geçersiz karar: 0
- Dondurulmuş karar CSV SHA-256:
  `711bfd27668ca1f64eba8627d0973cf27561b577d2d72871cbd16b89e7b3a173`
- Public review manifest SHA-256 değişmedi:
  `b551ef6bea86ee62c9fc831a48dd0b662a99344f25da7ec9ed550a2a27103bfb`
- Private manifest SHA-256 değişmedi:
  `b2b5da720ef256bc0c04a2aa788c4ff96de6ae372b1efd14997bbf45e9c4c462`
- Yeni dondurma testleri: 3/3 geçti
- Tam proje test paketi: 134/134 geçti

Karar kaynağı:

`research/candidate_generator_v2_5_visual_review_v1.csv`

## 4. Bu sonuç henüz neyi söylemez?

36 aynı-kaynak adayının candidate veya outside katmanında olduğunu henüz bilmiyoruz. Aynı şekilde
tek belirsiz çiftin hangi stratum'a ait olduğu açılmadı. Bu nedenle:

- v2.5 kabul edildi denemez;
- outside pozitif sayısı henüz raporlanamaz;
- private skorlar/stratumlar henüz yorumlanamaz;
- duplicate scorer, family construction, split ve model eğitimi başlatılamaz.

## 5. Sonraki hızlandırılmış kontrol noktası

Mehmet'in ayrı onayından sonra dondurulmuş karar CSV'si private selection manifestiyle tek seferlik
birleştirilecek. Önceden tanımlı bütün kabul kapıları hesaplanacak; `V25N000227` için gerektiğinde
ayrı adjudication yapılacak. Audit geçerse yeni deney tasarlamadan doğrudan duplicate scoring
çalıştırma planı ve ardından family-aware 70/15/15 split hazırlığına geçilecektir.
