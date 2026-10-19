# Multi Cancer veri seti — etiket anlamı denetimi

Tarih: 2026-10-15  
Durum: **Salt okunur kapsam ve etiket incelemesi; yeni model eğitimi yapılmadı.**

> Daha kapsamlı yerel metadata, özgün kaynak sayıları, kesin kopya sonuçları ve 26 etiketin tam
> kullanımı için `research/DATASET_MASTER_AUDIT_TR.md` ana denetim dosyasıdır.

## Kısa sonuç

Bu veri setindeki sekiz üst klasör, sekiz tutarlı "kanser tanısı" değildir. Bunlar farklı
kaynak veri setlerini ve çoğunlukla organ/görüntüleme gruplarını bir araya getirir. Bazı üst
klasörlerin içinde sağlıklı veya benign görüntüler de bulunur. Bu nedenle sekiz üst klasörle
eğitilen model:

- görüntünün sağlıklı mı kanserli mi olduğunu söylemez;
- kanserin alt türünü veya evresini söylemez;
- esas olarak görüntünün hangi üst veri seti/organ grubuna benzediğini söyler.

Örneğin `oral_normal` görüntüsünün sekiz sınıflı hedef etiketi `Oral Cancer`,
`kidney_normal` görüntüsünün etiketi `Kidney Cancer` olur. Bu, klasör adına göre teknik olarak
doğrudur fakat klinik anlamda "kanser teşhisi" değildir.

## Sekiz üst grubun içinde gerçekte ne var?

| Üst grup | Alt etiketler | Alt etiketin anlattığı şey | Kritik not |
|---|---|---|---|
| ALL | `all_benign`, `all_early`, `all_pre`, `all_pro` | Sağlıklı/benign örnek ve üç ALL hücre alt grubu | `early/pre/pro` adlarını kanserin ilerleme evreleri diye sunmamalıyız. |
| Brain Cancer | `brain_glioma`, `brain_menin`, `brain_tumor` | Glioma, meningioma ve pituitary tumor MRI görüntüleri | Normal beyin sınıfı yoktur; `brain_tumor` klasör adı aslında pituitary tumor anlamında kullanılmıştır. |
| Breast Cancer | `breast_benign`, `breast_malignant` | Benign veya malign meme histopatolojisi | Üst etiket tek başına maligniteyi söylemez. |
| Cervical Cancer | `cervix_dyk`, `cervix_koc`, `cervix_mep`, `cervix_pab`, `cervix_sfi` | Beş Pap smear hücre kategorisi | SIPaKMeD bunları normal, abnormal ve benign hücre kategorileri olarak tanımlar; beş kanser türü/evresi değildir. |
| Kidney Cancer | `kidney_normal`, `kidney_tumor` | Normal veya tümörlü böbrek CT görüntüsü | Kaynak veri setinde cyst ve stone da vardır; bu birleşik veri setine yalnızca normal/tumor alınmıştır. |
| Lung and Colon Cancer | `colon_aca`, `colon_bnt`, `lung_aca`, `lung_bnt`, `lung_scc` | İki ayrı organın benign ve malign histoloji etiketleri | Tek üst sınıfta hem akciğer hem kolon vardır. Üst sınıf maligniteyi veya organı tam olarak ayırmaz. |
| Lymphoma | `lymph_cll`, `lymph_fl`, `lymph_mcl` | Üç malign lenfoma alt türü | Normal/benign karşı sınıf yoktur. |
| Oral Cancer | `oral_normal`, `oral_scc` | Normal oral doku veya OSCC | Üst etiket tek başına kanser varlığını söylemez. |

Yerel `archive/README.md` bu 8/26 klasör yapısını ve veri artırma işlemini açıkça listeliyor.
Kaynak veri setleri de anlam farkını doğruluyor: beyin verisi üç MRI tümör tipinden, serviks
verisi normal/abnormal/benign Pap hücrelerinden, LC25000 ise iki organı kapsayan beş histoloji
sınıfından oluşuyor.

## Neden sekiz sınıflı yüksek doğruluk yanıltıcı olabilir?

Üst grupların görüntü türleri çok farklıdır: kan yayması mikroskopisi, beyin MRI, böbrek CT,
Pap smear ve çeşitli histopatoloji görüntüleri aynı hedef uzayında bulunur. Model, hastalık
özelliğinden önce bu görsel kaynak farklarını öğrenebilir. Böylece yüksek test doğruluğu şu
soruyu iyi cevaplayabilir:

> Bu görüntü hangi veri kaynağı/organ grubuna benziyor?

Fakat şu soruyu cevapladığını kanıtlamaz:

> Hastada kanser var mı; varsa türü veya evresi nedir?

Bu nedenle daha önceki sekiz sınıflı ResNet18 pilotunun `%97.08` accuracy ve `%97.02`
macro-F1 sonucu, yalnızca **üst veri seti grubu sınıflandırması için ön sonuç** olarak
saklanmalıdır. Klinik kanser teşhisi başarısı olarak kullanılmamalıdır.

## Olası proje hedefleri

### A — Yalnızca sekiz üst klasör

- Profesörün kayıtta tekrar ettiği "eight classes" ifadesine en doğrudan uyan görevdir.
- Teknik olarak kolay ve öğretim amacıyla geçerli bir görüntü sınıflandırma problemidir.
- Çıktı "kanser türü" değil, dürüstçe "üst veri seti/organ grubu" diye adlandırılmalıdır.
- Sağlıklı/kanserli veya alt tür bilgisi vermez.

### B — Tek modelle 26 alt klasör

- Veri setinde bulunan en ayrıntılı hazır etiketi tahmin eder.
- Normal/benign ve bazı hastalık alt türleri çıktıda ayrılır.
- Ancak 26 etiket aynı tıbbi ekseni paylaşmaz: bazıları organ, bazıları benign/malign,
  bazıları tümör tipi, bazıları hücre kategorisidir. Bu yüzden bunu da tek ve tutarlı bir
  "kanser evresi tahmini" diye sunamayız.

### C — Hiyerarşik yaklaşım

1. Görüntünün sekiz üst gruptan hangisine ait olduğunu bul.
2. O grubun içinde ilgili alt etiketi tahmin et.

Bu yaklaşım profesörün sekiz sınıf şartını korurken kullanıcıya daha anlamlı ikinci bir çıktı
verir. Örneğin `Oral Cancer -> oral_normal/oral_scc` veya
`Breast Cancer -> benign/malignant`. Yine de her grup için alt çıktının anlamı farklı olduğu
için raporda ortak bir "evre" iddiası yapılmaz.

### D — Tek bir sağlıklı/kanserli modeli

Mevcut 26 etiketten güvenilir biçimde doğrudan üretilemez. Bazı gruplarda açık normal/benign
etiketi vardır; fakat beyin ve lenfomada normal karşı sınıf yoktur, serviks kategorileri ikili
kanser/no-cancer etiketi değildir ve benign tümör ile malign kanseri eşitlemek tıbben doğru
değildir. Uzman tarafından onaylanmış yeni bir etiketleme olmadan bütün veri için ikili hedef
oluşturmayacağız.

## Önerilen yön

2026-10-15 çalışma kararı: Profesöre hemen soru göndermek yerine önce veri setinin sağlayabildiği
çıktıyı eksiksiz tanımlayacağız. İlk teknik aday, tek bir 26-alt-sınıf tahmini yapmak ve tahmin
edilen alt etiketten sekiz üst grup sonucunu da otomatik türetmektir. Böylece aynı deney hem tam
alt etiket hem de üst grup seviyesinde ölçülebilir. Önerilen teslim yapısı:

1. **Zorunlu temel:** Profesör isterse sekiz üst grubun sınıflandırılması; sonuç adı açıkça
   "top-level dataset-group classification" olur.
2. **Anlamlı genişletme:** 26 alt sınıf veya hiyerarşik üst-grup + alt-sınıf deneyi.
3. **Bilimsel kontrol:** Önceden artırılmış benzer görüntülerin train/test'e karışma denetimi,
   aynı bölmede adil model karşılaştırması ve sınıf bazlı metrikler.
4. **Sınır:** Model klinik tanı sistemi olarak sunulmaz; kanser evresi tahmini iddia edilmez.

Profesörle kapsam doğrulaması iptal edilmiş değildir; somut plan hazırlandıktan sonra ve final
deney/rapor iddialarından önce gerekirse yapılacaktır. Mehmet'in açık isteği olmadan mesaj
gönderilmeyecektir.

İleride gerekirse profesöre gönderilebilecek net soru:

> Hello Professor. I inspected the actual folder labels. The dataset has eight top-level
> folders, but some of them contain both normal/benign and tumor images—for example
> `kidney_normal/kidney_tumor`, `oral_normal/oral_scc`, and
> `breast_benign/breast_malignant`. Also, Lung and Colon are combined in one top-level folder.
> Should the required prediction be only the eight top-level folders, or should the project
> also predict the 26 subclasses (possibly hierarchically) so that normal/benign and disease
> subtypes are distinguished?

Bu soru, önceki soyut "8 mi 26 mı?" sorusundan farklıdır: README'deki somut etiket çelişkisini
gösterir ve yanlış bir tıbbi iddiayla aylarca ilerlememizi önler.

## Bu denetimde kullanılan kanıtlar

- Yerel `archive/README.md`: birleşik veri setinin klasörleri, kaynak bağlantıları ve augmentation
  ayarları.
- `transcript.pdf`: profesörün image classification, eight classes ve "what kind of cancer"
  ifadeleri; 26 alt sınıf veya sağlıklı/kanserli hedefi açıkça belirtilmiyor.
- Kaynak veri seti belgeleri: ALL, brain tumor, SIPaKMeD, LC25000, kidney, breast, lymphoma ve
  oral histopatoloji veri kaynakları.

