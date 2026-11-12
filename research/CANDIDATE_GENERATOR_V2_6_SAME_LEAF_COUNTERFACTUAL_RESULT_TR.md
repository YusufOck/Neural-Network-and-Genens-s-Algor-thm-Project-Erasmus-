# Candidate Generator v2.6 Same-Leaf Karşı-Olgusal Sonucu

Tarih: 2026-11-12
Durum: tam-artifact salt-okunur tanı tamamlandı; kapasite kapısı geçmedi

## Ne çalıştırıldı?

V2.5 aday birleşimi aynen korunarak yalnız Kidney CT ve Lymphoma histology gruplarında,
aynı 26-leaf etikete sahip robust rank 16–30 çiftleri eklendi. Mevcut 1.107 Drive
artifact'ı salt-okunur kullanıldı; duplicate scoring, family, split veya model eğitimi
yapılmadı.

## Doğrulanmış sonuç

- Rapor SHA-256: `1a3d1bedf5072633c69dc2335139a4beb19d6a41cc60dda203dadf21abf540f9`
- 132 ilgili artifact hash ile doğrulandı.
- Frozen top-15 tabanı birebir yeniden üretildi.
- Dört same-2D geliştirme referansının 4/4'ü bulundu.
- Kidney CT ek adayı: 112.337.
- Lymphoma histology ek adayı: 249.922.
- Toplam yeni aday: 362.259.
- Full aday toplamı: 3.357.129 = 25,824069 çift/görüntü.
- Frozen kapasite sınırı: 25 çift/görüntü = 3.250.000 çift.
- Sınır 107.129 çift aşıldı.

Kodun `RETURN CODE: 0` vermesi çalışmanın teknik olarak tamamlandığını gösterir. Deney
kapısı ise kapasite yüzünden başarısızdır. Bu nedenle v2.6 kabul edilmedi ve prospective
audit/family/split/training başlamadı.

## Sonraki dondurulmuş tanı

V2.6a, dört geliştirme çiftinin bilinen en küçük grup K değerlerini kullanarak üç sınırlı
profil karşılaştırır. Profil seçme kuralı sonuç görülmeden dondurulmuştur: 4/4 geliştirme
kapsamını ve 25 çift/görüntü kapasitesini geçen profiller arasında en geniş aday kapsamını
koruyan profil seçilir. Seçilen profil yalnız yeni path-disjoint prospective audit için
hipotez olabilir; geliştirme sonucu tek başına aday kuralını kabul ettiremez.
