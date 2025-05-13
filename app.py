import streamlit as st
from PIL import Image
from image_processor import ImageProcessor
from api_handler import APIHandler
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from env.example file
load_dotenv('env.example')

# Get API token from environment variables or use empty string as default
default_token = os.getenv("HUGGINGFACE_API_TOKEN", "")

# Kullanılabilecek modeller listesi
MODELS = {
    "ResNet-50": "microsoft/resnet-50",
    "ViT Base": "google/vit-base-patch16-224",
    "DeiT Base": "facebook/deit-base-distilled-patch16-224",
    "ConvNeXT": "facebook/convnext-base-224-22k-1k",
    "CLIP": "openai/clip-vit-base-patch32"
}

# Meyve listesi - model bu meyveleri tanıyabilir
MEYVELER = [
    "apple", "apricot", "avocado", "banana", "bell pepper", "blackberry", "blueberry", 
    "cantaloupe", "cherry", "coconut", "cucumber", "grape", "grapefruit", "kiwi", 
    "lemon", "lime", "mango", "orange", "papaya", "peach", "pear", "pineapple", 
    "plum", "pomegranate", "raspberry", "strawberry", "tomato", "watermelon"
]

# Meyve alt-türleri için eşleştirme tablosu
ALT_TURLER = {
    # Elma türleri
    "granny smith": "apple", 
    "red delicious": "apple",
    "golden delicious": "apple",
    "honeycrisp": "apple",
    "mcintosh": "apple",
    "braeburn": "apple",
    "gala": "apple",
    "pink lady": "apple",
    "fuji": "apple",
    # Portakal türleri
    "mandarin": "orange",
    "tangerine": "orange",
    "clementine": "orange",
    "satsuma": "orange",
    # Üzüm türleri
    "wine": "grape",
    "raisin": "grape",
    "sultana": "grape"
}

# Meyve Türkçe isim çevirileri
MEYVE_CEVIRILERI = {
    "apple": "Elma", "apricot": "Kayısı", "avocado": "Avokado", "banana": "Muz",
    "bell pepper": "Biber", "blackberry": "Böğürtlen", "blueberry": "Yaban Mersini",
    "cantaloupe": "Kavun", "cherry": "Kiraz", "coconut": "Hindistan Cevizi",
    "cucumber": "Salatalık", "grape": "Üzüm", "grapefruit": "Greyfurt",
    "kiwi": "Kivi", "lemon": "Limon", "lime": "Misket Limonu", "mango": "Mango",
    "orange": "Portakal", "papaya": "Papaya", "peach": "Şeftali", "pear": "Armut",
    "pineapple": "Ananas", "plum": "Erik", "pomegranate": "Nar", "raspberry": "Ahududu",
    "strawberry": "Çilek", "tomato": "Domates", "watermelon": "Karpuz"
}

def main():
    """
    Ana fonksiyon - Streamlit arayüzünü oluşturur
    """
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="Meyve Tanımlayıcı",
        page_icon="🍎",
        layout="wide"
    )
    
    # Başlık
    st.title("🍎 Meyve Tanımlayıcı")
    st.write("Bir meyve resmi yükleyin ve yapay zeka hangi meyve olduğunu tahmin etsin!")
    
    # Model seçimi
    selected_model_name = st.selectbox(
        "Kullanılacak model:",
        list(MODELS.keys())
    )
    selected_model_id = MODELS[selected_model_name]
    
    # API işleyicisini başlat (seçilen model ve token ile)
    api_handler = APIHandler(model_id=selected_model_id, api_token=default_token)
    
    # API yapılandırmasını kontrol et
    if not api_handler.is_configured():
        st.warning("⚠️ Hugging Face API anahtarı ayarlanmamış. Lütfen env.example dosyasına geçerli bir token ekleyin.")
    
    # Debug bilgisi göster
    with st.expander("Teknik Bilgiler"):
        st.info(f"Seçilen model: {selected_model_name} ({selected_model_id})")
        st.info(f"API Endpoint: {api_handler.api_url}")
        if len(default_token) > 10:
            st.info(f"Token: {default_token[:5]}...{default_token[-4:]}")
        else:
            st.warning("Token ayarlanmamış veya geçersiz")
    
    # İki sütun oluştur
    col1, col2 = st.columns([1, 1])
    
    # İlk sütunda dosya yükleyici
    with col1:
        st.subheader("Meyve Resmi Yükle")
        uploaded_file = st.file_uploader("Bir meyve resmi seçin...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Yüklenen resmi göster
                image = Image.open(uploaded_file)
                st.image(image, caption="Yüklenen Meyve", use_column_width=True)
                
                with st.expander("Görüntü Bilgileri"):
                    st.write(f"Boyut: {image.size}")
                    st.write(f"Format: {image.format}")
                    st.write(f"Mode: {image.mode}")
                
                # Tahmin butonu ekle
                if st.button("Meyveyi Tanımla", type="primary"):
                    if not api_handler.is_configured():
                        st.error("Lütfen önce API anahtarınızı env.example dosyasında yapılandırın.")
                    else:
                        with st.spinner(f"{selected_model_name} modeli ile analiz ediliyor..."):
                            # Resmi ön işle
                            processor = ImageProcessor()
                            processed_image = processor.preprocess(image)
                            
                            st.info("Görüntü işlendi ve API'ye gönderiliyor...")
                            
                            # Tahmin yap
                            prediction = api_handler.classify_image(processed_image)
                            
                            # Debug bilgisi
                            with st.expander("API Cevabı (Ham)"):
                                st.json(prediction)
            except Exception as e:
                st.error(f"Resim yüklenirken hata oluştu: {str(e)}")
    
    # İkinci sütunda sonuçlar
    with col2:
        if uploaded_file is not None and 'prediction' in locals():
            st.subheader("Meyve Tanımlama Sonuçları")
            
            if isinstance(prediction, dict) and "error" in prediction:
                st.error(prediction["error"])
                st.warning("API hatası oluştu. Lütfen token'ın doğru olduğundan emin olun ve tekrar deneyin.")
                
                # Hata yardımı
                st.info("Çözüm önerileri:")
                st.markdown("""
                1. Farklı bir model seçmeyi deneyin
                2. API token'ın doğru olduğundan emin olun
                3. Farklı bir görüntü yüklemeyi deneyin
                4. Görüntü formatını JPG olarak değiştirin
                """)
            else:
                # Sonucun boş olup olmadığını kontrol et
                if not prediction:
                    st.warning("API sonuç döndürmedi. Lütfen başka bir resim veya model deneyin.")
                else:
                    # Tüm sonuçları göster (meyve olmayanlar da dahil)
                    with st.expander("Tüm sınıflandırma sonuçları"):
                        for idx, result in enumerate(prediction[:10]):  # İlk 10 sonucu göster
                            label = result.get("label", "Bilinmeyen")
                            score = result.get("score", 0) * 100
                            st.write(f"{idx+1}. {label} ({score:.2f}%)")
                    
                    # Meyveleri filtrele ve sonuçları düzenle
                    meyve_sonuclari = []
                    for result in prediction:
                        # Değerleri kontrol et - label olmadığında veya None olduğunda hatayı önle
                        if "label" not in result or result["label"] is None:
                            continue
                            
                        # API'dan gelen etiketi küçük harfe çevir
                        original_label = result["label"]
                        label = original_label.lower()
                        score = result.get("score", 0) * 100
                        
                        # Eşleşen meyveyi bul
                        matched_fruit = None
                        
                        # 1. Önce alt türleri kontrol et
                        for alt_tur, ana_meyve in ALT_TURLER.items():
                            if alt_tur in label:
                                matched_fruit = ana_meyve
                                break
                                
                        # 2. Eğer alt tür bulunamadıysa, direkt meyve listesinde ara
                        if not matched_fruit:
                            for meyve in MEYVELER:
                                if meyve in label:
                                    matched_fruit = meyve
                                    break
                        
                        # 3. Eğer hala bulunamadıysa, bu kez etiketin meyve adı içerip içermediğini kontrol et
                        if not matched_fruit:
                            for meyve in MEYVELER:
                                if label in meyve:  # Örn: "apple" içinde "red" olabilir
                                    matched_fruit = meyve
                                    break
                        
                        # Eğer bir meyve bulunduysa, sonuçlara ekle
                        if matched_fruit:
                            turkce_isim = MEYVE_CEVIRILERI.get(matched_fruit, matched_fruit.capitalize())
                            meyve_sonuclari.append({
                                "Meyve": turkce_isim, 
                                "Eşleşme": f"{score:.2f}%", 
                                "Orijinal": original_label
                            })
                
                if meyve_sonuclari:
                    # Güzel bir sonuç gösterimi oluştur
                    st.success("Meyve başarıyla tanımlandı!")
                    
                    # Tahmin tablosunu göster
                    st.table(meyve_sonuclari[:5])  # En iyi 5 meyve tahmini
                    
                    # En iyi tahmini belirgin şekilde göster
                    top_meyve = meyve_sonuclari[0]["Meyve"]
                    top_score = meyve_sonuclari[0]["Eşleşme"]
                    st.markdown(f"### Bu bir **{top_meyve}** olabilir! ({top_score})")
                else:
                    st.warning("Bu resimde bilinen bir meyve tespit edilemedi. Lütfen başka bir meyve resmi deneyin.")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("Meyve Tanımlayıcı | Streamlit ve Hugging Face kullanılarak oluşturulmuştur")

if __name__ == "__main__":
    main() 