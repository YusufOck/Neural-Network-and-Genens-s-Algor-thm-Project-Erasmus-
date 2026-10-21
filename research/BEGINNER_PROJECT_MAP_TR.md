# NNGA projesi: başlangıç haritası ve şu anki çalışma

Bu belge, projeyi ilk kez açan Mehmet'in **hangi dosyanın nerede olduğunu, ne işe yaradığını ve
şu anda hangi adımda olduğumuzu** tek yerden görmesi için hazırlanmıştır.

## Şu an ne yapıyoruz?

Şu anda kanser sınıflandırma modeli eğitmiyoruz. Önce aynı kaynak görüntüden üretilmiş kopya ve
yakın-kopyaları bulma yöntemini doğruluyoruz. Amaç, daha sonra train/validation/test ayrımı
yapıldığında aynı görüntü ailesinin farklı bölümlere dağılıp data leakage oluşturmasını önlemektir.

Güncel sıra:

1. 278 belirsiz adayın önce yalnız **139 development** çifti gözle incelenecek.
2. ORB/RANSAC, ECC/phase correlation ve hizalanmış benzerlik kanalları kodlanacak.
3. Yöntem yalnız development sonuçlarıyla ayarlanıp dondurulacak.
4. O zamana kadar açılmayan 139 holdout çiftiyle yöntem bir kez ölçülecek.
5. Doğrulanmış yöntem bütün 26 alt sınıfa uygulanacak.
6. Yakın-kopya aileleri parçalanmadan 70/15/15 split hazırlanacak.
7. Ancak bundan sonra adil model eğitimi başlayacak.

## Proje klasör haritası

```text
NNGA/
├── README.md                         Projeye hızlı giriş ve doğrulanmış ana bilgiler
├── PROJECT_PLAN.md                   Baştan sona deney ve teslim planı
├── requirements.txt                 Python paketleri
├── config/
│   ├── label_taxonomy.json           26 alt sınıfın 8 üst gruba eşlenmesi
│   └── duplicate_audit_groups.json   5 görüntü modalitesi ve karşılaştırılabilir gruplar
├── src/
│   ├── data_index.py                 Veri kökünü bulur, 26 sınıfı indeksler
│   ├── audit_dataset.py              Sayı/boyut/format/bozuk görüntü denetimi
│   ├── audit_exact_duplicates.py     SHA-256 ile byte-byte aynı dosyaları bulur
│   ├── audit_perceptual_duplicates.py Eski iki-sınıflı pHash pilotu
│   ├── prepare_phash_review.py       Eski 120 pilot çifti için inceleme tablosu
│   ├── build_duplicate_calibration_set.py
│   │                                 26 sınıflı kalibrasyon setini üretir
│   ├── render_calibration_review.py  278 adayın 64 temas sayfasını üretir
│   ├── render_review_pair.py         Pair ID ile tek development çiftini büyük açar
│   └── train.py                      Model eğitim temeli; şu anda çalıştırılmayacak
├── research/                         İnsan tarafından okunabilir kanıt ve durum raporları
├── tests/                            Kodun beklenen davranışlarını otomatik kontrol eder
├── archive/                          Ham veri seti; değiştirilmez, GitHub'a gönderilmez
└── artifacts/                        Yerel audit çıktıları; GitHub'a gönderilmez
```

## Şu anda kullanacağımız üç yerel çıktı

### 1. Temas sayfaları

```text
artifacts/audit/duplicate_calibration_v1/review/contact_sheets/
```

64 PNG sayfası vardır. Her sayfada en çok altı çift bulunur. Sayfa genel tarama içindir.
Development sayfaları önce incelenir; holdout sayfaları yöntem dondurulana kadar açılmaz.

### 2. Pair ID indeksi

```text
artifacts/audit/duplicate_calibration_v1/review/contact_sheet_index.csv
```

Bir `C000...` Pair ID'nin hangi PNG sayfasında ve kaçıncı satırda olduğunu söyler. Bu bir karar
tablosu değildir.

### 3. İnsan karar tablosu

```text
artifacts/audit/duplicate_calibration_v1/review/review_labels.csv
```

278 çiftin asıl kayıt tablosudur. `human_verdict`, `confidence`, `review_notes`, `reviewer` ve
`review_date` alanları karar verilene kadar boş kalır. Boş satır **negatif** anlamına gelmez.

## Bir çifti kaybolmadan nasıl açacağız?

Örnek olarak `C000613`:

```powershell
$env:PYTHONUTF8="1"
py -3.10 .\src\render_review_pair.py --pair-id C000613
```

Oluşan dosya:

```text
artifacts/audit/duplicate_calibration_v1/review/pair_views/C000613.png
```

Araç yalnız development çiftini gösterir. Karar yazmaz, CSV'yi değiştirmez, eşik seçmez, split
oluşturmaz ve model eğitmez. Holdout çifti yanlışlıkla girilirse varsayılan olarak durur.

## Gözle karşılaştırma kontrol listesi

Amaç kanser tanısı koymak değildir. Soru yalnız şudur: **Bu iki dosya aynı kaynak görüntünün
dönüşmüş iki sürümü mü, yoksa birbirine benzeyen bağımsız görüntüler mi?**

1. Önce genel renge veya aynı sınıf adına bakarak karar verme.
2. Her iki görüntüde 3-5 nadir işaret seç:
   - aynı hücre/çekirdek kümeleri;
   - damar, boşluk veya bez sınırlarının tam biçimi;
   - doku yırtığı/kıvrımı;
   - lezyonun ve anatomik sınırların göreli konumu.
3. Parlaklık, küçük kayma, dönme, aynalama veya kırpma olabileceğini hesaba kat.
4. Tek bir geometrik dönüşümle bütün nadir işaretler aynı anda örtüşüyorsa
   `same_source_candidate` düşün.
5. Yalnız doku türü/renk benziyor fakat yerel işaretlerin konum ve şekilleri farklıysa
   `independent_lookalike` düşün.
6. Görüntü küçük, bulanık veya kırpılmış olduğu için güvenli karar verilemiyorsa `uncertain` seç.

Karar sözlüğü:

| Karar | Anlamı |
|---|---|
| `same_source_candidate` | Aynı kaynak görüntünün dönüşümleri olduğuna güçlü görsel kanıt var |
| `independent_lookalike` | Aynı sınıfa benzese de yerel yapılar bağımsız görüntüler gösteriyor |
| `uncertain` | Kanıt güvenli karar vermeye yetmiyor |

## Sentetik pozitif ne demektir?

Gerçek veri setinden bir kaynak görüntü seçilir; kod bu görüntünün parlaklığı değiştirilmiş,
döndürülmüş, aynalanmış veya kaydırılmış sürümünü üretir. İkinci görüntünün birinciden üretildiğini
bildiğimiz için çiftin doğru cevabı kesin olarak **aynı kaynak**tır. Buradaki “sentetik”, sahte bir
kanser dokusu üretildiği anlamına gelmez; yalnız test çiftinin ilişkisinin kontrollü biçimde
oluşturulduğunu belirtir.

Her alt sınıfta audit bölüm başına 2 kaynak × 10 dönüşüm = 20 sentetik pozitif vardır. Development
ve holdout birlikte alt sınıf başına 40; 26 alt sınıfta toplam 1.040 kontrollü sentetik pozitif
bulunur. Bunlar sınıflandırma modelinin eğitim verisi değildir; duplicate yönteminin neyi yakalayıp
kaçırdığını ölçmek içindir.

## pHash mesafesi neyi anlatır?

pHash her görüntüyü 64 bitlik kaba bir görsel parmak izine dönüştürür:

1. Görüntüyü gri tona çevirir ve 32×32 boyuta küçültür.
2. DCT ile geniş renk/parlaklık ayrıntısından çok genel uzamsal yapıyı frekans bileşenlerinde özetler.
3. Düşük frekanslı katsayıların medyanın altında/üstünde olmasına göre 64 adet 0/1 üretir.
4. İki parmak izindeki farklı bitler sayılır; buna Hamming mesafesi denir.

Mesafe 0-64 arasındadır. `0`, iki dosyanın byte-byte aynı olduğunu kanıtlamaz; yalnız pHash
özetlerinin aynı olduğunu söyler. Küçük değer genel yapının benzer olduğuna aday kanıt, büyük değer
ise pHash özetlerinin farklı olduğuna işarettir. Tek başına hiçbir değer kesin duplicate kararı
değildir.

Eski pilot yalnız örnekleme için 0-2, 3-4, 5-6, 7-8 ve 9-12 bantlarını kullandı. Bu, geçerli pHash
aralığının 0-12 olduğu veya 12'nin kesin eşik olduğu anlamına gelmez. 26 sınıflı kontrollü testte
sentetik pozitiflerin mesafesi 0-44 arasında çıktı:

- parlaklık değişimi çoğunlukla 0-10;
- yatay çevirme 26-36;
- %10 kaydırma 16-44;
- birleşik dönüşüm 20-40.

Parlaklık genel geometrik düzeni koruduğu için pHash çoğunlukla az değişir. Aynalama ve kaydırma
nesnelerin 32×32 özet içindeki yerini değiştirdiğinden çok sayıda bit değişebilir. Bu nedenle
`pHash <= 12` kuralı 1.040 bilinen akrabanın yalnız 488'ini (%46,92) tuttu, 552'sini (%53,08)
kaçırdı. Sonuç: pHash yararlı bir **aday üretme kanalıdır**, nihai karar yöntemi değildir.

## İlk birlikte inceleme: C000613

- Alt sınıf: `colon_aca` ↔ `colon_aca`
- Dosyalar: `colon_aca_0700.jpg` ↔ `colon_aca_4354.jpg`
- SHA-256 eşitliği: hayır
- pHash mesafesi: 10
- Şimdiki resmi insan kararı: boş

İlk görsel ön yorumda ortadaki koyu çekirdek kümesi, sağdaki yuvarlak boşluk, alttaki beyaz doku
yırtığı ve üst sınırdaki hücrelerin şekil/konumları iki tarafta birlikte örtüşüyor. Bu, yalnız aynı
sınıfa benzemekten daha güçlüdür ve `same_source_candidate` yönünde güçlü kanıttır. Ancak karar,
Mehmet ile ortak incelemeden sonra Pair ID üzerinden tabloya yazılacaktır.
