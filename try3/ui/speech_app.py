# Speech Recognition GUI — Authorized Speaker + Command Detection
# Run with: python -m streamlit run speech_app.py
# Requires: spk_model.pkl, cmd_model.pkl

import io
import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Speech Recognition System", layout="centered")

SAMPLE_RATE    = 16000
RECORD_SECS    = 1
BASIC_CONTROLS = ['on', 'off', 'stop', 'go', 'up', 'down']
LABEL_NAMES    = BASIC_CONTROLS + ['unknown']


#  MFCC extraction
def extract_mfcc_from_array(y_arr, sr=16000, n_mfcc=16, n_fft=512, hop_length=160, win_length=400):
    target_length = sr
    if len(y_arr) < target_length:
        y_arr = np.pad(y_arr, (0, target_length - len(y_arr)))
    else:
        y_arr = y_arr[:target_length]
    y_preemph = librosa.effects.preemphasis(y_arr)
    mfcc        = librosa.feature.mfcc(y=y_preemph.astype(float), sr=sr, n_mfcc=n_mfcc,
                                        n_fft=n_fft, hop_length=hop_length,
                                        win_length=win_length, window='hann')
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta_mfcc, delta2_mfcc])   # (48, ~100)


#  Two feature functions matching the notebook exactly
# def mfcc_to_features_spk(mfcc_matrix):
#     # Network 1: mean + std across time → (96,)
#     mean = mfcc_matrix.mean(axis=1)
#     std  = mfcc_matrix.std(axis=1)
#     return np.concatenate([mean, std])

def mfcc_to_features(mfcc_matrix):
    # Network 2: full flatten → (~4800,)
    return mfcc_matrix.flatten()


#  Recognition pipeline
def recognize_from_array(y_arr, spk_model, cmd_model, cmd_scaler):
    mfcc = extract_mfcc_from_array(y_arr)

    features_spk = mfcc_to_features(mfcc).reshape(1, -1)  # (1, 96)
    features_cmd = mfcc_to_features(mfcc).reshape(1, -1)  # (1, ~4800)
    features_cmd = cmd_scaler.transform(features_cmd)

    #  Speaker Gate (Network 1)
    spk_proba      = spk_model.predict_proba(features_spk)[0]
    spk_prediction = int(spk_model.predict(features_spk)[0])
    spk_confidence = float(spk_proba[1])
    is_authorized  = (spk_prediction == 1)

    #  Command Recognition (Network 2) (always runs regardless of authorization)
    cmd_proba      = cmd_model.predict_proba(features_cmd)[0]
    cmd_idx        = int(cmd_model.predict(features_cmd)[0])
    cmd_confidence = float(cmd_proba[cmd_idx])

    return {
        'authorized'     : is_authorized,
        'command'        : LABEL_NAMES[cmd_idx],
        'spk_confidence' : spk_confidence,
        'cmd_confidence' : cmd_confidence
    }


#  Model loader
@st.cache_resource
def load_models():
    base     = Path(__file__).parent
    spk_path = base / "spk_model.pkl"
    cmd_path = base / "cmd_model.pkl"
    cmd_scaler_path = base / "cmd_scaler.pkl"
    missing = [p.name for p in [spk_path, cmd_path, cmd_scaler_path] if not p.exists()]
    if missing:
        return None, None, missing

    return joblib.load(spk_path), joblib.load(cmd_path), joblib.load(cmd_scaler_path)


#  Result display
def show_result(result):
    st.markdown("---")

    # Always show the detected command
    st.markdown(f"### Command Detected: **`{result['command'].upper()}`**")

    if result['authorized']:
        st.success("AUTHORIZED")
    else:
        st.error("UNAUTHORIZED — Access denied.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speaker Confidence", f"{result['spk_confidence'] * 100:.1f}%")
        st.progress(result['spk_confidence'])
    with col2:
        st.metric("Command Confidence", f"{result['cmd_confidence'] * 100:.1f}%")
        st.progress(result['cmd_confidence'])


#  Main app
st.title("Speech Recognition System")
st.caption("Two-stage pipeline: Speaker Gate → Command Recognition")

spk_model, cmd_model, cmd_scaler = load_models()

if spk_model is None:
    st.error(f"Missing files: {', '.join(cmd_scaler)}")
    st.info(f"Looking in: `{Path(__file__).parent}`")
    st.stop()

st.success("Models loaded successfully")
st.markdown("---")

st.markdown("**Upload a `.wav` file to analyze.**")
st.info("Best results with 1-second clips at 16kHz, same format as training data.")

uploaded = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded is not None:
    st.audio(uploaded, format="audio/wav")

    if st.button("Analyze File", use_container_width=True):
        audio_bytes_io = io.BytesIO(uploaded.read())
        y_arr, _ = librosa.load(audio_bytes_io, sr=SAMPLE_RATE)

        with st.spinner("Processing..."):
            result = recognize_from_array(y_arr, spk_model, cmd_model, cmd_scaler)
        show_result(result)

st.markdown("---")
st.caption("Neural Network Based Speech Recognition")
