import streamlit as st
from PIL import Image
from models.model_loader import load_model
from utils.image_processing import preprocess_image
from utils.prediction import predict
from utils.descriptions import classes, descriptions, gejala, pencegahan, penanganan
import os
import numpy as np

# Ignore warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Print numpy version for debugging
print(f"Numpy Version: {np.__version__}")

# Set custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f5dc; /* Cream background */
        }
        .container {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #2E8B57;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 20px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }
        .description, .instruction {
            font-size: 18px;
            text-align: justify;
            color: #333333;
            margin-bottom: 20px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        button[data-baseweb="button"] {
            width: 100% !important;
            max-width: 300px;
            background-color: #2E8B57 !important;
            color: white !important;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px;
        }
        button[data-baseweb="button"]:hover {
            background-color: #228B22 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Home page
def home_page():
    st.markdown('<div class="container">', unsafe_allow_html=True)

    st.markdown('<div class="title">Dragonfruit Stem Health Detection App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Mendeteksi kesehatan batang buah naga secara otomatis</div>', unsafe_allow_html=True)

    # Menampilkan gambar dalam kolom di bawah subtitle
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("/Users/user/Documents/Kuliah Semester 5/STKI/Dragonfruit Detection/Image/12_jpg.rf.94ddd2899a58cc367ead4bc689e955e7.jpg", use_container_width=True, caption="Anthracnose")
    with col2:
        st.image("/Users/user/Documents/Kuliah Semester 5/STKI/Dragonfruit Detection/Image/1109__flipv__noise0-01_jpg.rf.ac81d84be7c7a1e6a51ae5846b69acba.jpg", use_container_width=True, caption="Cactusvirus")
    with col3:
        st.image("/Users/user/Documents/Kuliah Semester 5/STKI/Dragonfruit Detection/Image/14__flipv_jpg.rf.61b3407dc2bbe64ee8073ab913384181.jpg", use_container_width=True, caption="Health")
    with col4:
        st.image("/Users/user/Documents/Kuliah Semester 5/STKI/Dragonfruit Detection/Image/73_jpg.rf.ab96079daf58c809c519c51c89db5a53.jpg", use_container_width=True, caption="Steamcanker")

    st.markdown('<div class="description">Aplikasi ini dirancang untuk membantu petani dan peneliti dalam mendiagnosis kondisi batang buah naga dengan lebih cepat dan akurat. Menggunakan teknologi deep learning yang terlatih dengan data gambar batang buah naga yang terinfeksi berbagai penyakit, aplikasi ini memberikan solusi yang efisien untuk mengidentifikasi masalah kesehatan pada tanaman Anda.</div>', unsafe_allow_html=True)

    st.markdown('<div class="description">Dengan aplikasi ini, Anda dapat menganalisis gambar batang buah naga dan mengetahui apakah tanaman tersebut terinfeksi penyakit seperti Anthracnose, Cactusvirus, atau Steamcanker, atau apakah tanaman tersebut sehat. Semua proses ini dilakukan secara otomatis tanpa memerlukan pengetahuan teknis yang mendalam. Ini memungkinkan petani dan peneliti untuk menghemat waktu dan sumber daya dalam mengidentifikasi dan menangani masalah kesehatan tanaman.</div>', unsafe_allow_html=True)

    st.markdown('<div class="instruction"><strong>Panduan Penggunaan:</strong></div>', unsafe_allow_html=True)
    st.markdown("""
        <ul class="instruction">
            <li><strong>Klik tombol "Mulai Deteksi"</strong> untuk memulai proses deteksi.</li>
            <li><strong>Unggah gambar batang buah naga</strong> yang ingin Anda analisis, pastikan gambar cukup jelas agar hasil deteksi lebih akurat.</li>
            <li><strong>Tekan tombol "Deteksi"</strong> untuk memulai analisis gambar, dan lihat hasil prediksi mengenai kondisi batang buah naga Anda.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown('<div class="description"><strong>Manfaat Aplikasi:</strong></div>', unsafe_allow_html=True)
    st.markdown("""
        <ul class="instruction">
            <li><strong>Deteksi cepat dan akurat:</strong> Menggunakan model deep learning untuk analisis gambar batang buah naga dalam hitungan detik.</li>
            <li><strong>Penghematan waktu dan biaya:</strong> Petani dan peneliti dapat segera mengetahui kondisi tanaman tanpa perlu melakukan pemeriksaan manual yang memakan waktu.</li>
            <li><strong>Peningkatan hasil pertanian:</strong> Dengan identifikasi dini terhadap penyakit, tindakan pencegahan dan penanganan dapat dilakukan lebih cepat, meningkatkan produktivitas tanaman.</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Mulai Deteksi"):
        st.session_state.page = 'deteksi'
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
def detection_page():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<div class="title">Deteksi Kesehatan Batang Buah Naga</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Unggah gambar untuk memulai deteksi</div>', unsafe_allow_html=True)

    # Langkah 1: Panduan mengunggah gambar
    st.markdown("<div class='instruction'><strong>Langkah 1:</strong> Pilih gambar batang buah naga yang ingin dianalisis.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        # Langkah 2: Panduan untuk menekan tombol Deteksi
        st.markdown("<div class='instruction'><strong>Langkah 2:</strong> Tekan tombol 'Deteksi' untuk memulai analisis gambar.</div>", unsafe_allow_html=True)

        # Tombol Deteksi
        if st.button("Deteksi"):
            # Load model dan prediksi
            model_path = "resnet50_dragonfruitw_.pth"
            model = load_model(model_path)
            preprocessed_image = preprocess_image(image)
            class_name, confidence = predict(preprocessed_image, model, classes)

            # Menampilkan hasil prediksi dengan tampilan yang lebih rapi
            st.markdown(f"""
                <div style="margin-top: 20px;">
                    <div style="color: green; font-weight: bold; font-size: 20px;">Hasil Prediksi: {class_name}</div>
                    <div style="color: orange; font-size: 18px;">Kepercayaan: {confidence:.2f}</div>
                    <div style="margin-top: 10px; font-size: 16px;">
                        <strong>Gejala:</strong> {gejala[class_name]}<br>
                        <strong>Pencegahan:</strong> {pencegahan[class_name]}<br>
                        <strong>Penanganan:</strong> {penanganan[class_name]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Tombol Finish untuk mereset halaman ke beranda
            if st.button("Finish"):
                # Menyisipkan JavaScript untuk refresh halaman secara penuh
                st.markdown("""
                    <script type="text/javascript">
                        location.reload();
                    </script>
                """, unsafe_allow_html=True)

        else:
            # Memberikan instruksi jika tombol deteksi belum ditekan
            st.markdown("<div class='instruction'>Tekan tombol 'Deteksi' untuk melihat hasil analisis.</div>", unsafe_allow_html=True)
    else:
        # Instruksi jika gambar belum diunggah
        st.markdown("<div class='instruction'><strong>Catatan:</strong> Silakan unggah gambar terlebih dahulu untuk melakukan deteksi.</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Main function
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'deteksi':
        detection_page()

if __name__ == "__main__":
    main()