import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import io

TARGET_SR = 44100
TARGET_DURATION = 3
TARGET_LENGTH = TARGET_SR * TARGET_DURATION

@st.cache_resource
def load_assets():
    """Memuat model dan label encoder dari file."""
    try:
        model = load_model('model_ser_6823-68.h5') 
        data = joblib.load('label_data.joblib') 
        label_encoder = data['label_encoder']
        return model, label_encoder
    except Exception as e:
        st.error(f"Error saat memuat aset: {e}")
        return None, None

def preprocess_audio_from_bytes(audio_bytes, sr=TARGET_SR):
    try:
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio_trimmed) < TARGET_LENGTH:
            padded = np.pad(audio_trimmed, (0, TARGET_LENGTH - len(audio_trimmed)), mode='constant')
        else:
            padded = audio_trimmed[:TARGET_LENGTH]
        return padded
    except Exception as e:
        st.error(f"Error saat preprocessing audio: {e}")
        return None

def extract_log_mel(audio, sr=TARGET_SR):
    if audio is None: return None
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mean, std = np.mean(log_mel), np.std(log_mel)
    return (log_mel - mean) / std if std != 0 else log_mel - mean

st.set_page_config(page_title="Deteksi Emosi Suara", layout="wide")
st.title("🎤 Demo Deteksi Emosi dari Suara")
st.write("Unggah file audio (.wav atau .mp3) dan model akan mencoba mendeteksi emosi di dalamnya.")

model, label_encoder = load_assets()

if model and label_encoder:
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Deteksi Emosi"):
            with st.spinner("Sedang menganalisis audio..."):

                audio_bytes = uploaded_file.getvalue()
                
                processed_audio = preprocess_audio_from_bytes(audio_bytes)
                
                log_mel_spec = extract_log_mel(processed_audio)
                
                if log_mel_spec is not None:
                    
                    log_mel_spec_expanded = log_mel_spec[np.newaxis, ..., np.newaxis]

                    predictions = model.predict(log_mel_spec_expanded)
                    predicted_index = np.argmax(predictions, axis=1)[0]
                    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
                    
                    st.success(f"**Prediksi Emosi Terdeteksi:**")
                    st.metric(label="Emosi", value=predicted_emotion.capitalize())
                else:
                    st.error("Gagal mengekstrak fitur dari audio. Coba file lain.")
else:
    st.warning("Aset model tidak dapat dimuat. Pastikan file 'model_emosi.h5' dan 'preprocessed_data.joblib' ada di folder yang sama.")