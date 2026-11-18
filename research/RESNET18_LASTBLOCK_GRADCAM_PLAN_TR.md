# ResNet18 Last-Block Grad-CAM Deneyi — Dondurulmuş Plan

Tarih: 2026-11-18  
Durum: Kod ve örnek seçimi donduruldu; gerçek checkpoint/görüntü çalışması henüz yapılmadı.

## 1. Amaç

Bu çalışma, tamamlanmış `resnet18_lastblock_finetune_v1` modelinin seçili test görüntülerinde
kararını yaklaşık olarak hangi uzamsal bölgelerin etkilediğini Grad-CAM ile görselleştirir.
Yeni model eğitmez, model seçmez, test metriğini değiştirmez ve tıbbi lezyon lokalizasyonu
iddiasında bulunmaz.

Grad-CAM; hedef sınıf skorunun son convolution özellik haritalarına gradyanını hesaplar, bu
gradyanları kanal ağırlığı olarak kullanır ve pozitif etkilerin ağırlıklı toplamını bir ısı
haritasına dönüştürür. Bu projede hedef katman `layer4.1.conv2`'dir. 224×224 girdide bu katmanın
uzamsal çözünürlüğü düşüktür; bu nedenle harita kaba bir model-etki bölgesidir.

## 2. Değiştirilemeyen girdiler

- Model: yalnız `resnet18_lastblock_finetune_v1/best_model.pth`
- Checkpoint SHA-256: `538f2891075f0870e450cd6aa7eda0a5e06ecb73eefd14a4f6a0cd869759d54d`
- Split SHA-256: `999bf7a9bd1bba9f2957a9db0e13ae72adb3ecb8fdbc1f33cce40b083c1cf171`
- Test tahminleri SHA-256: `237f3283b88beeaf0a93f182adab23ad478810b0944245e3023e6cfe885c57bd`
- Görüntü: 224×224'e resize ve ImageNet normalizasyonu
- Örnek manifesti:
  `research/gradcam/resnet18_lastblock_gradcam_v1/sample_manifest.csv`
- Örnek manifesti SHA-256:
  `3f797979b4d75b092dfcb4e2348ea1705ed34d35da884ac999c68abef8c87add`

Kod, checkpointi yüklemeden ve görüntü açmadan önce bu hash'leri doğrulayacaktır.

## 3. Görüntü seçimi

Isı haritaları görülmeden önce 37 test görüntüsü donduruldu:

1. Her 26 leaf sınıftan bir doğru kontrol. Her sınıftaki doğru tahminlerin confidence medyanına
   en yakın örnek seçildi; eşitlikte alfabetik relative path kullanıldı.
2. İki cross-parent hatanın tamamı.
3. Confidence değeri en az 0,99 olan yedi leaf hatasının tamamı.
4. En sık karışan `oral_normal -> oral_scc` ve `oral_scc -> oral_normal` yönlerinin her birinden,
   o yöndeki hata confidence medyanına en yakın bir örnek.

Yollar tekrarsızdır. Bu tasarım, yalnız görsel olarak güzel veya beklenen bölgeleri vurgulayan
haritaları sonradan seçme riskini azaltır.

## 4. Üretilecek haritalar

- Doğru kontroller: tahmin edilen/gerçek sınıf için tek Grad-CAM haritası.
- Yanlış tahminler: yanlış tahmin edilen sınıf ve gerçek sınıf için iki ayrı Grad-CAM haritası.
- Her örnek: modelin gördüğü resize edilmiş görüntü, harita/overlay ve izlenebilir Sample ID.
- Toplu çıktı: bireysel PNG'ler, temas sayfaları, sonuç CSV'si, özet JSON ve hash manifesti.

Çalışma sırasında checkpointin seçili 37 görüntüdeki sınıf ve confidence değerleri yeniden
üretilmelidir. Sınıf farklıysa veya confidence mutlak farkı `5e-5` sınırını aşarsa işlem durur.

## 5. Yorum sınırları

- Grad-CAM nesne tespiti veya piksel-segmentasyonu değildir.
- Kırmızı/sarı alanlar yalnız modele etkisi yüksek yaklaşık bölgeleri gösterir.
- Haritanın dokuya bakması tıbbi doğruluk veya nedensellik kanıtı değildir.
- Model görüntü kenarı, arka plan veya veri-kaynağı artefaktına bakabilir.
- Bu post-hoc test audit'i yeni bir performans sonucu veya dış doğrulama değildir.
- Görseller ancak gerçek run tamamlanıp bütün hash ve yeniden-üretim kontrolleri geçtikten sonra
  rapora alınacaktır.

## 6. Uygulama ve çalıştırma

- Analiz kodu: `src/run_gradcam_audit.py`
- Dondurulmuş config: `config/gradcam_resnet18_lastblock_v1.json`
- Colab giriş noktası: `notebooks/resnet18_lastblock_gradcam_colab.ipynb`
- Planlanan Drive çıktısı:
  `/content/drive/MyDrive/NNGA_OUTPUTS/resnet18_lastblock_gradcam_v1`

Bu aşama `post_hoc_explainability_audit` olarak sınıflandırılır; eğitim, ablation veya final
klinik deney değildir.
