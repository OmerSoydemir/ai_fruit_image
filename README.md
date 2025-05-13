# 🍎 Meyve Tanımlayıcı

Yapay zeka destekli meyve tanımlama uygulaması. Meyve resmini yükleyin, yapay zeka hangi meyve olduğunu tahmin etsin!

![Meyve Tanımlayıcı](https://img.shields.io/badge/Meyve%20Tan%C4%B1mlay%C4%B1c%C4%B1-v1.0-green) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-API-yellow)

## ✨ Özellikler

- **🖼️ Kolay Resim Yükleme**: Bilgisayarınızdan kolayca meyve resmi yükleyin
- **🔄 Çoklu Model Desteği**: 5 farklı yapay zeka modeli arasından seçim yapın
  - ResNet-50 (Microsoft)
  - ViT Base (Google)
  - DeiT Base (Facebook)
  - ConvNeXT (Facebook)
  - CLIP (OpenAI)
- **🔍 Detaylı Sonuçlar**: Sonuçları güven skorlarıyla birlikte görüntüleyin
- **🇹🇷 Türkçe Destek**: Meyve adları Türkçe olarak gösterilir
- **🐞 Hata Ayıklama Araçları**: Gelişmiş teknik bilgilere erişim

## 📋 Gereksinimler

- Python 3.8 veya daha yüksek
- Streamlit 1.28.0+
- Pillow (PIL) 10.1.0+
- Requests 2.31.0+
- Hugging Face Hub
- python-dotenv 1.0.0+

## 🚀 Kurulum

1. Bu repoyu klonlayın:
   ```bash
   git clone https://github.com/kullanici-adiniz/meyve-tanimlayici.git
   cd meyve-tanimlayici
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Hugging Face API anahtarınızı ayarlayın:
   - [Hugging Face](https://huggingface.co/settings/tokens) adresinden bir token alın
   - `env.example` dosyasını açın ve `your_token_here` yerine kendi token'ınızı yazın:
     ```
     HUGGINGFACE_API_TOKEN=your_token_here
     ```

## 🔧 API Bağlantısının Test Edilmesi

Uygulamayı başlatmadan önce, API bağlantınızı test etmek için:

```bash
python test_api.py
```

Bu komut, token'ınızın doğru çalışıp çalışmadığını kontrol eder ve yapılandırma sorunlarını giderir.

## 💻 Kullanım

1. Uygulamayı başlatın:
   ```bash
   streamlit run app.py
   ```

2. Web tarayıcınızda uygulama açılacaktır (genellikle http://localhost:8501)

3. Kullanım adımları:
   - Soldaki model seçiciden bir yapay zeka modeli seçin
   - "Meyve Resmi Yükle" bölümünden bir resim yükleyin
   - "Meyveyi Tanımla" butonuna tıklayın
   - Sonuçları sağ tarafta görüntüleyin

4. **İpuçları**:
   - En iyi sonuçlar için net ve yalnızca meyve içeren fotoğraflar kullanın
   - Model farklı sonuçlar veriyorsa, başka bir model deneyin
   - "Teknik Bilgiler" bölümünden detaylı API cevaplarını görebilirsiniz

## 🔍 Desteklenen Meyveler

Uygulama şu meyveleri tanıyabilir:
- 🍎 Elma (çeşitli türleri dahil: Granny Smith, Red Delicious, vb.)
- 🍌 Muz
- 🍊 Portakal ve Turunçgiller
- 🍋 Limon
- 🥝 Kivi
- 🍇 Üzüm
- 🍓 Çilek
- 🍑 Şeftali
- 🥭 Mango
- ... ve 20+ farklı meyve türü daha!

## 🧩 Teknik Detaylar

### Mimari

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Streamlit   │     │ Image         │     │ Hugging Face    │
│ Web Arayüzü │────▶│ Processor     │────▶│ Inference API   │
└─────────────┘     └───────────────┘     └─────────────────┘
       ▲                    │                      │
       │                    │                      │
       └────────────────────┴──────────────────────┘
                          Sonuçlar
```

### Çalışma Prensipleri

1. **Görüntü Ön İşleme**:
   - Görüntü doğrulanır ve geçerliliği kontrol edilir
   - 224×224 veya model için uygun boyuta dönüştürülür
   - RGB formatına dönüştürülür

2. **API İletişimi**:
   - İşlenmiş görüntü Hugging Face API'sine gönderilir
   - 4 farklı gönderim yöntemi otomatik denenir (binary, base64, vb.)
   - Modelin döndürdüğü sonuçlar doğrulanır ve işlenir

3. **Sonuç İşleme**:
   - API sonuçları meyve listesiyle eşleştirilir
   - Özel alt türler ana meyve türlerine eşlenir (örn. "Granny Smith" → "Elma")
   - Türkçe karşılıkları ve eşleşme oranları gösterilir

## 📊 Performans Karşılaştırması

| Model       | Doğruluk | Hız  | Tavsiye Edilen Kullanım                 |
|-------------|----------|------|----------------------------------------|
| ResNet-50   | Yüksek   | Hızlı| Genel amaçlı meyve tanıma              |
| ViT Base    | Çok Yüksek| Orta | Detaylı tür tanıma                     |
| DeiT Base   | Yüksek   | Hızlı| Hızlı tanıma gereken durumlar          |
| ConvNeXT    | Çok Yüksek| Yavaş| Zorlu ışık koşullarında               |
| CLIP        | Orta     | Orta | Daha geniş yelpazede sınıflandırma     |

## 🐛 Sorun Giderme

**Yaygın Sorunlar ve Çözümleri:**

1. **API Hatası**: 
   - API token'ınızın doğru ve geçerli olduğundan emin olun
   - Başka bir model seçmeyi deneyin

2. **Meyve Tanımlanamıyor**:
   - Daha net bir fotoğraf kullanın
   - Arka planı sade olan bir görüntü deneyin
   - Başka bir model seçin

3. **Uygulama Başlatma Sorunu**:
   - Tüm bağımlılıkların kurulu olduğunu doğrulayın
   - `requirements.txt` ile tekrar kurulum yapın

## 🤝 Katkıda Bulunma

1. Bu repo'yu fork edin
2. Yeni bir branch oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin yeni-ozellik`)
5. Pull Request açın

## 📜 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için LICENSE dosyasına bakın.

## 🙏 Teşekkürler

- Önceden eğitilmiş modeller için [Hugging Face](https://huggingface.co/)
- Web uygulama çerçevesi için [Streamlit](https://streamlit.io/)
- Fikir ve geri bildirimler için tüm katkıda bulunanlar 