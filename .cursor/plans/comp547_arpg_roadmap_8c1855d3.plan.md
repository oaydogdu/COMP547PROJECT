---
name: COMP547 ARPG Roadmap
overview: ARPG fikrini küçük ölçekte güvenilir şekilde yeniden üretip, tek-GPU/Colab Pro kısıtı altında hız-kalite tradeoff’unu sistematik olarak haritalayan bir proje planı.
todos:
  - id: scope-freeze
    content: Reproduction + controlled empirical analysis kapsamını ve minimum deney setini kilitle
    status: completed
  - id: baseline-build
    content: Küçük ölçekli autoregressive baseline eğitim/örnekleme hattını çalışır hale getir
    status: completed
  - id: rpd-integrate
    content: Randomized parallel decoding mekanizmasını ekle ve correctness testleri yap
    status: completed
  - id: run-ablations
    content: Paralellik ve schedule ablation deneylerini maliyet kontrollü şekilde çalıştır
    status: completed
  - id: evaluate-report
    content: FID + hız metrikleriyle tradeoff eğrilerini üret ve final anlatıyı oluştur
    status: completed
isProject: false
---

# COMP547 Projesi Yol Haritası (ARPG Speed–Quality Tradeoff)

## Part Part Uygulama Planı (Genel ve Kullanılabilir)

### Part 1 — Çerçeveyi Kilitle
- Projenin tek cümle hedefini yaz: hız-kalite dengesinde ARPG ne zaman faydalı?
- Başarı ölçütlerini sabitle: FID, latency, throughput.
- Minimum teslim kapsamını sabitle: Fashion-MNIST + CIFAR-10, sınırlı ama düzenli deney matrisi.

### Part 2 — Baseline Sistemi Kur
- Klasik autoregressive üretimi çalışan hale getir.
- Fashion-MNIST üzerinde eğitim + örnekleme sanity check yap.
- Pipeline’ın tekrarlanabilir çalıştığını (seed/log/checkpoint) doğrula.

### Part 3 — ARPG Mekanizmasını Entegre Et
- Randomized parallel decoding ekle.
- Küçük ve kısa koşularda correctness/stabilite testleri yap.
- Baseline ile aynı koşullarda ilk hız farkını gözlemle.

### Part 4 — Kontrollü Pilot Deneyler
- Küçük bir pilot deney matrisi çalıştır (az kombinasyon, kısa koşu).
- 3 ekseni hızlıca yokla: parallelism, schedule, dataset zorluğu.
- Umut veren ayarları seç, zayıf ayarları erken ele.

### Part 5 — Ana Deneyleri Çalıştır
- Öncelik T4 ile maliyet kontrollü ana sweep yap.
- Gerekirse sadece kritik koşullar için daha güçlü GPU kullan.
- Her koşulda aynı ölçüm protokolüyle hız + kalite sonuçlarını topla.

### Part 6 — Analiz ve Sonuç Çıkarımı
- Speed-quality tradeoff eğrilerini üret.
- “Hangi ayar aralığı pratikte mantıklı?” sorusunu net cevapla.
- Randomized schedule’ın hangi veri zorluğunda avantaj/dezavantaj verdiğini yorumla.

### Part 7 — Rapor ve Sunum Paketleme
- Katkıyı doğru konumlandır: yeni mimari değil, pratik fayda sınırlarının analizi.
- Ana bulguları 3 başlıkta sun: ne işe yaradı, nerede bozuldu, bütçe dostu öneri.
- Final teslim için grafik, tablo ve örnek görselleri tek anlatıda birleştir.

### Part 8 — Zaman Kalırsa (Opsiyonel)
- Üçüncü, daha zor bir küçük dataset ekle.
- En iyi 1-2 ayarda ek tekrarlarla güven aralığını güçlendir.
- Gelecek iş olarak daha büyük ölçek veya farklı decoding stratejisi öner.

## Hedef ve Başarı Kriterleri
- **Ana hedef:** ARPG’nin randomized parallel decoding fikrini küçük ölçekte doğrulamak ve hangi koşullarda pratik fayda sağladığını göstermek.
- **Birincil katkı türü:** Dengeli yaklaşım (reproduction + kontrollü ampirik analiz), ancak final anlatıda **tradeoff karakterizasyonu** bir adım daha önde.
- **Başarı kriterleri:**
  - Tek-GPU ortamında (öncelik T4) tekrarlanabilir deney pipeline’ı
  - En az bir güçlü baseline’a karşı anlamlı latency/throughput kazanımı
  - Kalite düşüşünün paralellik/schedule/dataset karmaşıklığı eksenlerinde net raporlanması

## Proje Kapsamı
- **Kapsama dahil:**
  - Küçük ölçekli AR model (discrete token üretimi) ve randomized parallel decoding
  - Fashion-MNIST (debug) + CIFAR-10 (ana benchmark)
  - 3 eksenli analiz: parallelism seviyesi, decoding schedule, veri karmaşıklığı
- **Kapsam dışı (şimdilik):**
  - ImageNet ölçeği
  - Yeni mimari önerisi (odak: yöntem analizi)
  - Çok büyük hiperparametre taramaları

## Teknik Strateji (Maliyet Duyarlı)
- **Varsayılan hesaplama profili:** Colab Pro + T4.
- **A100 kullanım kuralı:** Yalnızca kritik noktalarda (final run, dar boğaz doğrulaması, tekrarlanması pahalı ablation).
- **Maliyet kontrol prensipleri:**
  - Önce küçük çözünürlük/az epoch ile yön doğrulama
  - Erken durdurma ve düşük varyanslı, kısa pilot deneyler
  - Uzun koşuları sadece hipotez doğrulandıktan sonra çalıştırma

## Deney Tasarımı
- **Aşama 1 — Reproduction çekirdeği:**
  - Oto-regresif baseline’ı çalıştır
  - Random-order training + block-parallel decoding mekanizmasını entegre et
  - Küçük veri ve kısa koşularda stabilite/sanity kontrolü
- **Aşama 2 — Kontrollü tradeoff analizi:**
  - Eksen A: paralellik seviyesi (ör. block size / simultaneous tokens)
  - Eksen B: decoding schedule (random vs sabit/structured alternatifler)
  - Eksen C: dataset karmaşıklığı (Fashion-MNIST → CIFAR-10)
- **Aşama 3 — Sonuç paketleme:**
  - Speed–quality eğrileri
  - Nitel örnek karşılaştırmaları
  - “Ne zaman faydalı, ne zaman değil?” karar rehberi

## Ölçüm ve Raporlama Protokolü
- **Kalite metrikleri:** FID + nitel örnekler.
- **Verimlilik metrikleri:** sample latency, throughput (img/s), mümkünse GPU memory footprint.
- **Adil karşılaştırma kuralları:**
  - Aynı model checkpoint’i ile farklı decoding stratejileri
  - Aynı batch/seed politikasına mümkün olduğunca sadık ölçüm
  - Warm-up sonrası ölçüm penceresi
- **İstatistiksel güven:** En az 3 tekrar (kritik deneylerde), ortalama + std raporu.

## Haftalık Yol Haritası (Revize)
- **Hafta 1:** Kod tabanı seçimi/adaptasyonu, baseline eğitim ve örnek üretim pipeline’ı.
- **Hafta 2:** Randomized parallel decoding entegrasyonu, correctness testleri, ilk hız ölçümleri.
- **Hafta 3:** Paralellik seviyesi ablation’ı (ana sweep), ilk tradeoff eğrileri.
- **Hafta 4:** Schedule ablation’ı + CIFAR-10 ana deneyleri.
- **Hafta 5:** Ek tekrarlar, grafiklerin finalize edilmesi, bulguların yazıya dökülmesi.
- **Hafta 6:** Sunum ve final rapor polishing (sınırlılıklar, tehditler, gelecek iş).

## Riskler ve Önleyici Plan
- **Risk:** FID hesaplama maliyeti yüksek.
  - **Önlem:** Pilot aşamada küçük örnek sayısı, finalde tam ölçüm.
- **Risk:** Randomized decoding kaliteyi erken bozabilir.
  - **Önlem:** Paralellik seviyesini kademeli artırma, schedule kontrollü kıyas.
- **Risk:** Colab oturum kesintileri.
  - **Önlem:** Sık checkpoint, deney log standardı, yeniden başlatma scriptleri.
- **Risk:** Deney matrisi aşırı büyüme.
  - **Önlem:** Ön-kayıtlı minimum deney seti + sadece umut veren kollarda genişleme.

## Minimum Deney Seti (Teslim Garantisi)
- Fashion-MNIST: baseline + 2 paralellik seviyesi + 2 schedule.
- CIFAR-10: baseline + 3 paralellik seviyesi + 2 schedule.
- Her koşul için hız ölçümü + FID + örnek grid.

## Karar Verme Çerçevesi (Finalde Cevaplanacak Sorular)
- Hangi paralellik aralığında hız kazancı kalite kaybına değiyor?
- Random schedule her zaman daha mı iyi, yoksa veri karmaşıklığına mı bağlı?
- Tek-GPU öğrenci bütçesinde en iyi “cost/performance” reçetesi nedir?

## Sonraki Adım
- Bu plan onaylanınca birlikte önce **deney altyapısı şablonunu** (config, logging, evaluation script yapısı) netleştirip, ardından minimum deney setini adım adım uygularız.