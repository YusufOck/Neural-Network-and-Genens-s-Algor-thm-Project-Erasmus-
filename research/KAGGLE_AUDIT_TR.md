# Kaggle Multi Cancer — İlk İnceleme Notları

Tarih: 2026-10-14  
Durum: **Salt okunur inceleme. Bu turda hiçbir üçüncü taraf notebooku çalıştırılmadı.**

> 2026-10-15 kapsam düzeltmesi: Bu dosyadaki "8 ana kategori" ifadesi sekiz kanser tanısı
> anlamına gelmez. Bunlar farklı kaynak/organ gruplarıdır ve bazıları normal/benign alt etiketler
> içerir. Ayrıntılı eşleme için `research/LABEL_SEMANTICS_AUDIT_TR.md` dosyasına bakın.

## İnceleme soruları

Her çalışma için şu soruları soruyoruz:

1. Gerçek hedefi 8 ana kategori mi, 26 alt sınıf mı, yoksa tek organın alt türleri mi?
2. Hangi model kullanılıyor?
3. Model sıfırdan mı, ImageNet ağırlıklarıyla mı başlıyor?
4. Train/validation/test nasıl ayrılıyor?
5. Gerçek bağımsız test seti var mı?
6. Accuracy dışında macro-F1 ve sınıf bazlı sonuçlar var mı?
7. Önceden artırılmış benzer görüntüler bölümler arasında karışabilir mi?
8. Kod yerelde yeniden üretilebilir mi ve hangi düzeltmeler gerekir?

## Data sayfası

- 8 ana kategori, 26 alt sınıf, toplam 130.000 JPEG görüntü.
- Her alt sınıfta 5.000 görüntü bulunuyor; görüntüler 512×512.
- Kaynaklar farklı görüntü türlerini karıştırıyor: MRI, CT, histoloji ve mikroskopi.
- README'ye göre veri daha önce rotation, shift, shear, zoom, horizontal flip ve brightness
  değişimleriyle artırılmış.

## İncelenen Kaggle notebookları

| Çalışma | Gerçek hedef | Model | Bölme | Raporlanan sonuç | İlk karar |
|---|---|---|---|---|---|
| Multiple Cancer Classification - VGG16 | Her ana klasör için ayrı 2–5 alt sınıf modeli | ImageNet VGG16, dondurulmuş gövde | %70 train / %30 validation | Organa göre %83.40–%98.77 validation accuracy | Profesörün sekiz üst-grup taban çizgisiyle aynı görev değil; validation yanlış biçimde test diye adlandırılıyor |
| Cancer classification 97.7 accuracy | Yalnızca ALL klasöründeki 4 lösemi alt sınıfı | 64×64 özel CNN, sıfırdan | %75 train / %20 validation / %5 test | %97.70 test accuracy, yaklaşık 0.98 macro-F1 | Başlık yanıltıcı; 8 ana kategori sonucu değil |
| Multi-Cancer_Classification_Pytorch | Sekiz ayrı grup modeli; her biri kendi alt sınıflarını tahmin ediyor | ImageNet ResNet18 | Yaklaşık %80/%10/%10, rastgele | Gruba göre %84.2–%98.8 test accuracy | Profesörün sekiz üst-grup taban çizgisiyle aynı görev değil; seed yok ve olasılık kodunda sorun var |

## Discussion sayfası

Yalnızca iki konu var ve ikisinin de yanıtı yok:

- Meme histolojisi görüntülerinin büyütme oranı sorulmuş.
- Serviks görüntülerinde kullanılan augmentation tekniği sorulmuş.

Dolayısıyla Discussion bölümü büyütme oranını, hasta/grup kimliklerini veya artırılmış görüntülerin
hangi orijinal görüntüden geldiğini açıklamıyor.

## Şimdilik öğrendiğimiz en önemli şey

Kaggle başlığındaki yüksek accuracy ile bizim proje hedefimizin aynı olduğunu varsayamayız. Önce
etiket hedefini ve veri bölme yöntemini okumalıyız. Bu üç popüler çalışma, tek bir 8-ana-kategori
modeli kurmuyor.

## Sonraki inceleme adımı — henüz çalıştırılmadı

Önce 8/26 veya hiyerarşik hedef profesörle netleştirilecek. Ardından Mehmet'in onayıyla seçilen
hedefe gerçekten uyan adayları Code aramasıyla filtreleyeceğiz. Bir adayı çalıştırmadan önce hücre
hücre inceleyip yerel makineye uyarlamak için gereken değişiklikleri çıkaracağız.
