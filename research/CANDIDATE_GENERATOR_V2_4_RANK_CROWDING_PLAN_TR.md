# N000123 tam-indeks rank-crowding tanı planı

Tarih: 2026-11-06

Durum: **Mehmet tarafından onaylandı; kod/config tam sonuç görülmeden hazırlanıp
hash-kilitlendi. Tam Drive artifact değerlendirmesi henüz çalıştırılmadı.**

Deney türü: **Salt-okunur, predeclared, tam-indeks odaklı tanı**

Config SHA-256:
`c5073b99fcafcbfc87e691a8250d818ffc39ac2f7de8c8635e3c9c7bba867ab7`

Analyzer SHA-256:
`bcdafcb1c3a1df1d6b464c1fa4578ab3de1237ca6bb6eb06470e5008edab4484`

## Amaç

V2.4 tam-artifact ablation'ında doğal aynı-kaynak referansları 22/23 kaldı. Eksik
`N000123`, 6.604-görüntülük audit indeksinde robust union rank 1 olmasına rağmen tam
130.000 görüntülük v2.3/v2.4 birleşiminde yoktur. pHash mesafesi 34 olduğu için
`pHash<=12` güvenlik ağına da girmez.

Bu tanı şu iki soruyu cevaplayacaktır:

1. N000123'ün dört frozen robust descriptor kanalında, tam 25.000-görüntülük
   `cervix_pap_smear` havuzundaki iki yönlü sıraları nedir?
2. Serviks robust top-K değeri yalnızca önceden belirlenen
   `15, 20, 25, 30, 40, 50, 75, 100` değerlerine çıkarılırsa, serviks ve bütün
   130.000 görüntü için exact aday yükü ne olur?

## Girdiler ve değişmezler

- Drive'daki değiştirilmeyen v2.3 output klasörü ve 1.107 kayıtlı artifact.
- Run-contract SHA-256:
  `6c4884d25da3f5d2af2b6d64522b3a78af08db5e05e9ce99e4b13d12686a5261`.
- 25.000 serviks indeks satırı, beş serviks descriptor artifact'ı, dört frozen
  robust-top15 artifact'ı ve 196 serviks pHash block artifact'ı.
- Önceden doğrulanan v2.4 yükü: 2.491.553 aday.
- Serviks baseline: 355.588 robust top-15, 68.770 pHash ve 378.508 birleşim adayı.

Kod, hesapladığı K=15 robust birleşimini frozen dört robust artifact'ın gerçek
birleşimiyle birebir karşılaştıracaktır. K=15 için bütün-veri projeksiyonu 2.491.553
değilse rapor yazılmayacaktır.

## Hesap yöntemi

Her descriptor kanalı için 25.000×512 normalize descriptor okunur. Bloklu matris
çarpımıyla her görüntünün en çok ilk 100 komşusu, frozen retrieval ile aynı stable
tie-break kuralıyla sıralanır. Yönlü komşuluklar canonical çiftlere çevrilir ve her
çift için dört kanal/iki yön içindeki en iyi sıra saklanır.

Her K için:

- robust çiftler `minimum_rank <= K` ile seçilir;
- mevcut serviks `pHash<=12` çiftleriyle birleşim alınır;
- doğrulanmış v2.4 toplamından eski serviks birleşimi çıkarılıp yeni birleşim eklenir;
- toplam çift/görüntü ve 25 kapasite sınırı raporlanır.

Bu çalışma descriptor üretmez ve görüntü ZIP'ine ihtiyaç duymaz. T4 yalnız matris
çarpımını hızlandırır. Yaklaşık 200 MB descriptor okunur; dört 25.000×25.000 arama
nedeniyle birkaç ila onlarca dakika sürebilir. Log ve son rapor Drive'a ayrı yazılır.

## Bu tanının yapmayacağı işler

- V2.3 artifact'larını değiştirmez.
- V2.5 kuralı seçmez veya dondurmaz.
- Duplicate scorer çalıştırmaz.
- Family oluşturmaz.
- Train/validation/test split hazırlamaz.
- Model eğitmez.

Tanı raporu birlikte incelendikten sonra ayrı onayla bir v2.5 hipotezi önerilebilir.
