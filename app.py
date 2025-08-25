import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import io
from audio_recorder_streamlit import audio_recorder

TARGET_SR = 44100
TARGET_DURATION = 3
TARGET_LENGTH = TARGET_SR * TARGET_DURATION
MAX_RECORD_DURATION = 6 

@st.cache_resource
def load_assets():
    try:
        model = load_model('model_ser_cnn-91.h5') 
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

def validate_audio_duration(audio_bytes):
    """Validasi durasi audio maksimal 6 detik"""
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes))
        duration = len(audio) / sr
        return duration <= MAX_RECORD_DURATION, duration
    except:
        return False, 0

def process_recorded_audio(audio_bytes):
    """Proses audio yang direkam dan potong jika lebih dari 6 detik"""
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR)
        
        max_samples = int(MAX_RECORD_DURATION * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            st.info(f"⚠️ Audio dipotong menjadi {MAX_RECORD_DURATION} detik")
        
        import soundfile as sf
        from io import BytesIO
        
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio, sr, format='WAV')
        audio_buffer.seek(0)
        processed_bytes = audio_buffer.getvalue()
        audio_buffer.close()
        
        return processed_bytes, len(audio) / sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, 0

def analyze_emotion(audio_bytes):
    """Fungsi untuk menganalisis emosi dari audio bytes"""
    processed_audio = preprocess_audio_from_bytes(audio_bytes)
    log_mel_spec = extract_log_mel(processed_audio)
    
    if log_mel_spec is not None:
        log_mel_spec_expanded = log_mel_spec[np.newaxis, ..., np.newaxis]
        predictions = model.predict(log_mel_spec_expanded)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
        
        return predicted_emotion, predictions[0]
    return None, None

st.set_page_config(page_title="Deteksi Emosi Suara", layout="wide")
st.title("🎤 Demo Deteksi Emosi dari Suara")
st.write("Rekam suara langsung atau unggah file audio (.wav atau .mp3) untuk mendeteksi emosi.")

model, label_encoder = load_assets()

if model and label_encoder:
    tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Rekam Suara"])
    
    with tab1:
        st.subheader("Upload File Audio")
        st.write(f"Maksimal durasi: {MAX_RECORD_DURATION} detik")
        
        uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            is_valid, duration = validate_audio_duration(audio_bytes)
            
            if not is_valid:
                st.error(f"⚠️ File audio terlalu panjang! Durasi: {duration:.2f} detik. Maksimal {MAX_RECORD_DURATION} detik.")
            else:
                st.success(f"✅ File valid. Durasi: {duration:.2f} detik")
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("🔍 Deteksi Emosi dari File", key="detect_upload"):
                    with st.spinner("Sedang menganalisis audio..."):
                        predicted_emotion, predictions = analyze_emotion(audio_bytes)
                        
                        if predicted_emotion is not None:
                            st.success("**Hasil Deteksi Emosi:**")
                            st.metric(label="Emosi", value=predicted_emotion.capitalize())
                            
                            with st.expander("📊 Confidence Scores"):
                                emotions = label_encoder.classes_
                                for emotion, score in zip(emotions, predictions):
                                    st.progress(float(score), text=f"{emotion.capitalize()}: {score:.3f}")
                        else:
                            st.error("Gagal mengekstrak fitur dari audio. Coba file lain.")
    
    with tab2:
        st.subheader("Rekam Suara Langsung")
        st.write(f"Klik tombol mikrofon dan rekam suara Anda (akan otomatis dipotong jika lebih dari {MAX_RECORD_DURATION} detik)")
        
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        if 'audio_processed' not in st.session_state:
            st.session_state.audio_processed = None
        if 'audio_duration' not in st.session_state:
            st.session_state.audio_duration = 0
        
        audio_bytes = audio_recorder(
            text="Klik untuk mulai merekam",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,  
            energy_threshold=(-1000),
        )
        
        if audio_bytes and audio_bytes != st.session_state.recorded_audio:
            st.session_state.recorded_audio = audio_bytes
            processed_bytes, duration = process_recorded_audio(audio_bytes)
            st.session_state.audio_processed = processed_bytes
            st.session_state.audio_duration = duration
        
        if st.session_state.recorded_audio and st.session_state.audio_processed:
            st.success("✅ Rekaman berhasil!")
            st.info(f"📊 Durasi rekaman: {st.session_state.audio_duration:.2f} detik")
            
            st.audio(st.session_state.audio_processed, format='audio/wav')
            
            if st.button("🔍 Deteksi Emosi", key="detect_record"):
                with st.spinner("Sedang menganalisis rekaman..."):
                    predicted_emotion, predictions = analyze_emotion(st.session_state.audio_processed)
                    
                    if predicted_emotion is not None:
                        st.success("**Hasil Deteksi Emosi:**")
                        st.metric(label="Emosi", value=predicted_emotion.capitalize())
                        
                        with st.expander("📊 Confidence Scores"):
                            emotions = label_encoder.classes_
                            for emotion, score in zip(emotions, predictions):
                                st.progress(float(score), text=f"{emotion.capitalize()}: {score:.3f}")
                    else:
                        st.error("Gagal mengekstrak fitur dari rekaman.")
        elif st.session_state.recorded_audio is None:
            st.info("🎙️ Belum ada rekaman audio. Klik tombol mikrofon untuk mulai merekam.")

    with st.expander("ℹ️ Informasi Aplikasi"):
        st.write(f"""
        **Fitur Aplikasi:**
        - 📁 Upload file audio (WAV/MP3) dengan validasi durasi maksimal {MAX_RECORD_DURATION} detik
        - 🎙️ Rekam suara langsung dengan pemotongan otomatis jika melebihi {MAX_RECORD_DURATION} detik
        - 🔍 Deteksi emosi menggunakan model CNN
        - 📊 Tampilan confidence scores untuk semua kategori emosi
        
        **Catatan:**
        - Pastikan browser mengizinkan akses microphone
        - File audio yang diupload akan divalidasi durasinya
        - Rekaman akan otomatis dipotong jika melebihi {MAX_RECORD_DURATION} detik
        - Model bekerja optimal dengan audio berkualitas baik
        """)

else:
    st.warning("Aset model tidak dapat dimuat. Pastikan file 'model_ser_cnn-91.h5' dan 'label_data.joblib' ada di folder yang sama.")
