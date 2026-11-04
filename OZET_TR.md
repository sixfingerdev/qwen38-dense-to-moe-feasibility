# SixFinger Qwen3.8-27B Dense-to-MoE Deneyinin Teknik Özeti

**Durum:** Çalışma sonuçlandırıldı. Hedeflenen kalite korumasını ve hız/bellek kazanımını doğrulayan dönüştürülmüş bir model yayımlanmıyor.

## Amaç

Bu çalışmanın amacı, `Qwen/Qwen3.8-27B` modelinin yoğun (dense) SwiGLU ileri beslemeli katmanlarını herhangi bir gradyan tabanlı eğitim, fine-tuning veya distillation işlemi uygulamadan seyrek uzman kümelerine (Mixture-of-Experts, MoE) dönüştürmenin uygulanabilirliğini araştırmaktı. Başarı için üç koşul birlikte gerekliydi:

1. Asıl modelin pratik dil anlama, akıl yürütme ve kod üretme kalitesinin neredeyse korunması.
2. Gerçek çıkarım sırasında GPU belleği (VRAM) tepe kullanımının azalması.
3. Aynı donanım ve yük altında gerçek token üretim hızının artması.

Sadece FFN ağırlıklarını gruplara ayırmak bu koşulları sağlamıyor. Ayrıca yalnızca tek bir katmanın matematiksel çıktısı üzerinde yapılan ölçüm, modelin genel başarısının ölçümü sayılamaz.

## Yöntem

İlk aşamada dense FFN'nin `gate`, `up` ve `down` projeksiyonlarına ait ara nöron kanalları, birbirleriyle kesişmeyen uzman gruplarına ayrıldı. Tüm gruplar hesaplandığında orijinal FFN çıktısının kayan nokta toplama farkları dışında yeniden elde edilebildiği gösterildi.

Ardından hesaplama yükünü düşürmek amacıyla kanalların bir bölümünün çalıştırılmadığı yaklaşımlar incelendi. Öne çıkan eğitim gerektirmeyen stratejiler şunlardı:

- Kalibrasyon verilerinden çıkarılan sabit önem sıralaması.
- Çalıştırılmayan kanalların ortalama katkısını önceden hesaplayarak telafi etme.
- Yerel Jacobian ve doğrusal olmayan artık katkılarını yaklaşık temsil etme.
- PCA ile sıkıştırılmış bir girdiden, tam ara aktivasyonları hesaplamadan önem sıralaması tahmin etme.

Deneyler, orijinal Qwen ağırlıklarının yalnızca **ilk decoder FFN katmanına** odaklandı. Kodlama ağırlıklı örneklerden alınmış **91 token'lık kalibrasyon** ve farklı örneklerden alınmış **47 token'lık test** aktivasyonları kullanıldı. Modelin ağırlıkları değiştirilmedi; gradyanla eğitim yapılmadı.

## Sayısal bulgular

| Çalıştırılan FFN ara kanalları | Sabit seçim + ortalama telafisi, en iyi | PCA rank-80 seçici | Tam aktivasyona bakan oracle |
| ---: | ---: | ---: | ---: |
| %25 | %41,845 | %43,335 | %29,123 |
| %50 | %26,254 | %29,346 | %10,128 |
| %75 | %13,949 | %16,595 | %2,092 |

Bu değerler, ilk FFN çıktısının orijinal dense çıktıya göre **ortalama göreli L2 hata oranlarıdır**. Modelin MMLU, kodlama veya başka bir testte aynı yüzdeyle başarısız olduğu anlamına gelmez. Bu tür bir uçtan uca kalite testi yapılmadı.

Son sütundaki oracle, hangi kanalların önemli olduğunu belirlemek için zaten bütün ara aktivasyonları hesaplıyor. Bu yüzden bu seçici gerçek çıkarımda hesaplamayı azaltan uygulanabilir bir yönlendirici değil; yalnızca karşılaştırma üst sınırıdır.

## Önemli teknik sonuçlar

**1. Kayıpsız bölme ve seyrek hesaplama aynı şey değildir.** Tüm uzmanlar çalıştırılırsa dense FFN'yi yeniden üretmek mümkündür; ancak hesaplama maliyeti esasen devam eder.

**2. Çalıştırılmayan uzmanların katkısı sıfır değildir.** Ortalama telafisi ve düşük-rank tahminleri bazı durumlarda hatayı azaltabilse de test edilen oranlarda kayıpsız seviyeye indirmedi.

**3. Düşük hesap maliyetli doğru seçim kritik darboğazdır.** Rank-80 PCA yaklaşımı küçük kalibrasyon setindeki girdi varyansının yaklaşık %99,6'sını açıklamasına rağmen doğru kanalları seçmekte oracle seviyesine yaklaşamadı.

**4. VRAM azalması gösterilmedi.** Bütün uzman ağırlıkları GPU'da tutuluyorsa bunları yeniden adlandırmak veya parçalara ayırmak bellek tüketimini azaltmaz. Uzmanların CPU/RAM üzerinde saklanması ayrı bir aktarım maliyeti oluşturur.

**5. Uçtan uca hızlandırma gösterilmedi.** Daha az nöron hesaplayan tek bir deneysel FFN çekirdeğinin mikrosaniye düzeyindeki performansı, 27B modelin gerçek token/s hızına doğrudan çevrilemez.

## Yayın kararı

Deney, başarısızlığın da bilimsel sonuç olduğuna dayanarak yayımlanmıştır. Ne tam ağırlıklı yeni bir MoE checkpoint'i, ne genel model kalitesini koruduğuna dair bir sonuç, ne de doğrulanmış VRAM/token hız kazanımı bulunmaktadır. Bunlar varmış gibi sunmak yanlış olur.

Çalışma, dense-to-MoE dönüşümün özellikle **eğitimsiz, ucuz ve doğru yönlendirme** bölümündeki teknik zorluğu belgeliyor. Bu bulgular bütün olası yaklaşımların başarısız olacağını kanıtlamaz; yalnızca uygulanan yöntemlerin belirtilen hedefi karşılamadığını gösterir.

Orijinal model: https://huggingface.co/Qwen/Qwen3.8-27B  
Deney defteri (hazırlık tarihinde özel; erişim gerekebilir): https://www.kaggle.com/code/antigravityiin/sixfinger-qwen3-8-moefication-lab/edit/run/365661914  
Ayrıntılı İngilizce rapor: [`REPORT.md`](REPORT.md)  
Makine tarafından okunabilir ölçümler: [`results/real_first_ffn_holdout.csv`](results/real_first_ffn_holdout.csv)
