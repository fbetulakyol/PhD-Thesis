# Methodology Review: NIH ChestX-ray14 (CXR) → MosMedData (CT), Unsupervised Domain Adaptation

**Kısa cevap:** Deney, *kapsamı doğru tanımlanırsa* unsupervised cross-modality domain
adaptation olarak sunulabilir. Ancak bu, standart UDA'nın temel varsayımının
(kaynak ve hedefte aynı görev, değişen yalnızca P(x)) **ihlal edildiği** bir senaryodur.
Modalite farkı (covariate shift) yanında **label shift** ve kısmi **concept shift** de vardır.
Bu durum deneyi geçersiz kılmaz. Yine de tezde iddia
"CXR'den CT'ye COVID-19 tanısının transferi" olarak değil,
**"CXR'de öğrenilen parankimal opasite/anormallik tespitinin CT'ye aktarılabilirliği"**
olarak kurulmalıdır.

| # | Konu | Tespit | Koddaki karşılığı |
|---|------|--------|-------------------|
| 1 | **Label semantiği** | NIH `Pneumonia`/`Infiltration`, raporlardan NLP ile çıkarılmış, gürültülü ve heterojen etiketlerdir (bakteriyel/viral pnömoni, ödem, atelektazi vb.). MosMed CT-1..4 ise yalnızca **COVID-19 viral pnömonisinin yaygınlık derecesidir** (buzlu cam ağırlıklı, periferik, bilateral). Ortak payda "akciğer parankim opasitesi"dir. Hastalık sınıfı birebir aynı değildir, yani concept shift vardır. | Pozitif tanım parametrik (`INCLUDE_INFILTRATION`). Ablation olarak ikisini de raporlayın. |
| 2 | **Infiltration** | ChestX-ray14'ün en düşük etiket güvenilirliğine sahip sınıfıdır ve klinikte artık önerilmeyen bir terimdir. Pozitif sayısını ~15 kat artırır, ama hedef semantiği zayıflatır. | `INCLUDE_INFILTRATION=True/False` iki ayrı koşu olarak raporlanmalı. |
| 3 | **Negatif sınıflar** | `No Finding` genel hastane popülasyonudur (yaşlı, yatan hasta). `CT-0` ise COVID şüphesiyle taranan ve CT'de viral pnömoni bulgusu olmayan hastalardır, başka patoloji taşıyabilirler. "Pnömoni yok" ortak, ama popülasyon ve dışlama kuralı farklıdır. Ayrıca NIH'de diğer bulgulardan yalnızca birini taşıyan görüntüler dışlanır, pozitiflerde ise eşlik eden bulgular kalır. Bu da kısayol (shortcut) riski yaratır. | Dışlama kuralı ve "eşlik eden bulgulu pozitif" sayısı loglanır. |
| 4 | **Label shift** | NIH'de pozitif oranı yaklaşık %26 (Infiltration ile) veya yaklaşık %2 (yalnız Pneumonia). MosMed'de yaklaşık %77. Marjinal hizalama yapan DANN, label shift altında **sınıfları yanlış eşleyebilir** (Zhao et al., 2019, "On learning invariant representations"). Bu, negative transfer'in teorik bir nedenidir. | Prevalans her split'te yazdırılır. Sabit 0.5 eşiği anlamsızlaşır, bu yüzden validation'da belirlenen eşik raporlanır. |
| 5 | **2D projeksiyon ile 3D kesit farkı** | CXR, tüm toraksın üst üste binmiş bir projeksiyonudur. Bir CT kesiti ise ince bir aksiyal kesittir. Yapısal bir eşleşme yoktur. Ek olarak **slice-level label gürültüsü** vardır: CT-1'de (<%25 tutulum) pozitif bir volume'ün birçok kesiti lezyonsuzdur. Volume etiketi, kesit etiketi gibi davranmaz. | Eğitimde merkezi %30–70 bölgesinden rastgele kesit. Değerlendirmede 7 eşit aralıklı kesitin ortalaması alınır, metrikler volume düzeyindedir. `VOLUME_AGGREGATION="max"` ablation olarak eklendi. |
| 6 | **Mean agregasyon** | Mean, fokal (CT-1) hastalığı seyreltir ve sensitiviteyi düşürür. Protokol gereği varsayılan mean kalmalıdır, ama tezde max ile karşılaştırma önerilir. | `VOLUME_AGGREGATION` |
| 7 | **Target validation etiketi** | MosMed VAL AUC ile checkpoint seçmek, egitim gradyanına etiket sokmasa da, **saf UDA değildir** (Musgrave et al., 2021; "UDA model selection" literatürü). Metinde "küçük etiketli target validation ile UDA" olarak raporlanmalıdır. Validation seti küçüktür (~110 volume, ~25 negatif), bu yüzden seçim gürültülüdür. | `USE_TARGET_LABELS_FOR_MODEL_SELECTION`. `False` iken MosMed etiketleri test anına kadar **hiç okunmaz**, loglanmaz bile. |
| 8 | **Unsupervised seçim yanlılığı** | Source validation loss, DANN için erken epoch'ları (λ küçükken) seçme eğilimindedir, yani daha az adaptasyona doğru yanlıdır. | `UNSUPERVISED_SELECTION = "source_val_loss" \| "last_epoch"` |
| 9 | **Negative transfer** | Modalite farkı çok büyük olduğu için domain classifier ayrımı neredeyse trivialdir. GRL, backbone'u bilgiyi silmeye zorlayabilir. Domain acc'nin %50'ye inmesi başarı kanıtı değildir: zayıf bir domain head de %50 verir. | Domain acc (source ve target ayrı), λ, NIH-val AUC her epoch loglanır. NIH-test AUC tabloda yer alır, böylece kaynak performansının çöküşü görünür. |
| 10 | **MMD/CORAL yorumu** | DANN feature normunu küçülterek MMD/CORAL'ı "yapay" olarak düşürebilir. | Metrikler L2-normalize feature'larda hesaplanır (ham değerler detailed CSV'de). Ayrıca lineer domain-probe accuracy / proxy-A-distance raporlanır. |
| 11 | **NIH data leakage (eski kodda vardı)** | NIH'de ~112k görüntü ~30k hastaya aittir. Görüntü düzeyinde split, aynı hastayı train ve val'e koyar. Bu, kaynak metriklerini ve `source_val_loss` seçimini şişirir. | **Hasta düzeyinde** stratified split (`StratifiedGroupKFold`) ve overlap assert'i. |
| 12 | **BatchNorm tuzağı (eski kodda vardı)** | Source ve target'ı ayrı forward'larla vermek, eğitimde BN'in her domain'i kendi istatistiğiyle normalize etmesine yol açar. Bu örtük bir AdaBN'dir, DANN'in parçası değildir. Eval'deki running stats ile de tutarsızdır. | Tek forward'da `cat([xs, xt])` kullanılır. |
| 13 | **İstatistiksel güç** | MosMed TEST ~222 volume ve tek seed var. AUC farkları ±0.03–0.05 mertebesinde gürültülüdür. | Bootstrap %95 CI ve DANN − Source Only için paired bootstrap p-değeri. Tezde en az 3–5 seed ile ortalama ± std raporlayın. |
| 14 | **Referans noktası** | Source Only ile DANN arasındaki fark tek başına domain gap'in büyüklüğünü göstermez. | Opsiyonel **Target Oracle** (MosMed TRAIN etiketleriyle supervised). **UDA değildir**, yalnızca üst sınır olarak raporlanır. |
| 15 | **CT ön işleme** | NIfTI orientasyonu çalışmadan çalışmaya farklı olabilir. Ayrıca rastgele kesit seçimini "path → kesit" şeklinde cache'lemek hatalıdır. | `as_closest_canonical` ile RAS orientasyonu kullanılır. NaN/−Inf değerleri HU_MIN'e çekilir, ardından lung window uygulanır. Cache **volume** düzeyindedir (merkezi slab, uint8, mmap), kesit her `__getitem__` çağrısında yeniden çekilir. |

**Önerilen deney matrisi (tez için):**
`{Source Only, DANN, Target Oracle} × {INCLUDE_INFILTRATION ∈ {True, False}} × {selection ∈ {target_val_auc, source_val_loss}} × seed ∈ {42, 43, 44}`.
İleri adım olarak label shift'e dayanıklı yöntemler (CDAN, MDD veya class-balanced/importance-weighted DANN) ve
MosMed CT-0'ın tek başına anlamlı bir "normal" sınıf olup olmadığına dair sınırlılık tartışması önerilir.
