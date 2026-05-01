# Speech Recognition GUI — Authorized Speaker + Command Detection
# Run with: python -m streamlit run speech_app.py
# Requires: spk_model.pkl, cmd_model.pkl (saved from the notebook)


# We use it to create BytesIO objects — essentially a file that lives in RAM
# instead of on disk, so we never need to write temp files.
import io
# Every UI element (buttons, text, tabs, etc.) is created by calling st.something().
import streamlit as st
import numpy as np
import librosa

# Joblib is used to save and load Python objects to/from disk.
# We use it to load the trained sklearn MLPClassifier models from .pkl files.
import joblib

# Sounddevice lets Python talk to the computer's microphone and speakers.
# We use it to record live audio from the mic.
import sounddevice as sd
# We use it to encode our recorded audio array into WAV format for playback.
import scipy.io.wavfile as wav
from pathlib import Path


st.set_page_config(
    page_title="Speech Recognition System",
    layout="centered"
)


SAMPLE_RATE = 16000
# The model was trained on 1-second clips, so we record exactly 1 second.
RECORD_SECS = 1
BASIC_CONTROLS = ['on', 'off', 'stop', 'go', 'up', 'down']
# The index of each word here corresponds to the integer the model outputs.
# So if the model predicts 2, that maps to LABEL_NAMES[2] = 'stop'.
LABEL_NAMES = BASIC_CONTROLS + ['unknown']


#MFCC Extraction 
def extract_mfcc_from_array(y_arr, sr=16000, n_mfcc=16, n_fft=512, hop_length=160, win_length=400):

    target_length = sr

    if len(y_arr) < target_length:
        # Audio is too short — pad the end with zeros (silence) to reach 1 second.
        y_arr = np.pad(y_arr, (0, target_length - len(y_arr)))
    else:
        # Audio is too long — slice off everything after 16000 samples.
        y_arr = y_arr[:target_length]

    mfcc = librosa.feature.mfcc(
        y=y_arr.astype(float),
        sr=sr,
        n_mfcc=n_mfcc,         
        n_fft=n_fft,          
        hop_length=hop_length, 
        win_length=win_length, 
        window='hann'         
    )

    delta_mfcc = librosa.feature.delta(mfcc)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, delta_mfcc, delta2_mfcc])


def mfcc_to_features(mfcc_matrix):

    mean = mfcc_matrix.mean(axis=1)
    std = mfcc_matrix.std(axis=1)
    return np.concatenate([mean, std])


#Recognition Pipeline
def recognize_from_array(y_arr, spk_model, cmd_model):
    # Step 1: Extract MFCC features from the raw audio array.
    # Result shape: (48, ~100)
    mfcc = extract_mfcc_from_array(y_arr)

    # Step 2: Convert the MFCC matrix to a flat feature vector.
    # Result shape: (96,)
    # .reshape(1, -1) turns (96,) into (1, 96) — sklearn models expect a 2D array
    features = mfcc_to_features(mfcc).reshape(1, -1)

    # Speaker Recognition (Network 1) 
    # predict_proba returns the probability for each class.
    # For the binary speaker model, shape is (1, 2): [p_unauthorized, p_authorized].
    # [0] takes the first (and only) row, giving array [p_unauth, p_auth].
    spk_proba = spk_model.predict_proba(features)[0]

    # predict() returns the predicted class label: 0 (unauthorized) or 1 (authorized).
    # [0] takes the first element. int() converts from numpy int to plain Python int.
    spk_prediction = int(spk_model.predict(features)[0])

    # spk_proba[1] is the probability of class 1 (authorized).
    spk_confidence = float(spk_proba[1])

    is_authorized = (spk_prediction == 1)

    #Command Recognition (Network 2) 
    # We always run this regardless of authorization — the GUI shows the command
    #even for unauthorized speakers, so the user can see what was said.

    # predict_proba returns 7 probabilities, one per command class.
    # Shape: (1, 7) → after [0]: (7,)
    cmd_proba = cmd_model.predict_proba(features)[0]

    # predict() returns the index of the predicted class (0 through 6).
    cmd_idx = int(cmd_model.predict(features)[0])

    # The confidence for the predicted class specifically (not all 7).
    # cmd_proba[cmd_idx] picks out just the probability for the winning class.
    cmd_confidence = float(cmd_proba[cmd_idx])

    # Convert the integer class index to its string label.
    predicted_cmd = LABEL_NAMES[cmd_idx]

    # Return a dictionary with all results.
    # The caller accesses values like result['authorized'], result['command'], etc.
    return {
        'authorized':    is_authorized,
        'command':       predicted_cmd,
        'spk_confidence': spk_confidence,
        'cmd_confidence': cmd_confidence
    }


# ─── Model Loader ─────────────────────────────────────────────────────────────
# @st.cache_resource is a decorator — it wraps the function below with extra behavior.
# Streamlit reruns the entire script on every user interaction. Without caching,
# the models would be reloaded from disk on every button click (very slow).
# @st.cache_resource runs the function once, keeps the result in memory, and
# returns the cached result on every subsequent call without re-executing the function.
@st.cache_resource
def load_models():
    spk_path = Path("spk_model.pkl")
    cmd_path = Path("cmd_model.pkl")


    if not spk_path.exists() or not cmd_path.exists():
        return None, None

    # joblib.load() deserializes the .pkl file back into the sklearn MLPClassifier object.
    # Returns both models as a tuple — the caller unpacks them into two variables:
    # spk_model, cmd_model = load_models()
    return joblib.load(spk_path), joblib.load(cmd_path)


# ─── Result Display ───────────────────────────────────────────────────────────
def show_result(result):
    st.markdown("---")
    cmd = result['command']

    # Render a large heading with the detected command.
    # **...** in Markdown = bold. `...` in Markdown = monospace/code font.
    st.markdown(f"### Command Detected: **`{cmd.upper()}`**")
    
    # st.success renders a styled green banner. st.error renders a red one.
    if result['authorized']:
        st.success("AUTHORIZED")
    else:
        st.error("UNAUTHORIZED — Access denied, but command was still detected.")

    # Returns two column objects we assign to col1 and col2.
    col1, col2 = st.columns(2)

    # Everything indented under "with col1:" is placed inside the left column
    with col1:
        # st.metric renders a styled number card with a label above and value below.
        st.metric("Speaker Confidence", f"{result['spk_confidence'] * 100:.1f}%")
        # st.progress renders a horizontal progress bar.
        st.progress(result['spk_confidence'])

    # Everything indented under "with col2:" is placed inside the right column
    with col2:
        st.metric("Command Confidence", f"{result['cmd_confidence'] * 100:.1f}%")
        st.progress(result['cmd_confidence'])


# Main App 

# Render the page title (largest heading) and a small grey subtitle below it.
st.title("Speech Recognition System")
st.caption("Two-stage pipeline: Speaker Gate -> Command Recognition")

spk_model, cmd_model = load_models()

# If either model file was missing, load_models() returned None
# We show an error message and call st.stop() which halts the rest of the script
# so nothing below this runs 
if spk_model is None:
    st.warning("""
    Models not found.
    Place spk_model.pkl and cmd_model.pkl in the same folder as this script.
    """)
    st.stop()

# If we reach here, both models loaded successfully.
st.success("Models loaded successfully")

# Another horizontal divider for visual separation.
st.markdown("---")


# Input Mode Tabs 
# st.tabs() creates a tabbed interface and returns one object per tab.
# The user can click between tabs without the page reloading.
tab1, tab2 = st.tabs(["Record Audio", "Upload WAV File"])


# ── Tab 1: Record ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("**Record a 1-second voice command.**")

    # st.info renders a blue info box.
    st.info("Say one of: `on`  ·  `off`  ·  `stop`  ·  `go`  ·  `up`  ·  `down`")

    # st.session_state is a dictionary that persists across reruns.
    # Normal Python variables reset every rerun because Streamlit re-executes
    # the whole script. session_state survives reruns, so we use it to store
    # the recorded audio between the "Record" click and the "Analyze" click.
    # This if block initializes the key only once on the very first run.
    if 'recording' not in st.session_state:
        st.session_state.recording = None

    # Create two side-by-side columns with a 2:1 width ratio.
    col_rec, col_clear = st.columns([2, 1])

    with col_rec:
        # st.button renders a clickable button.
        # It returns True on the one rerun where the user just clicked it, False otherwise.
        record_btn = st.button("Start Recording (1 sec)", use_container_width=True)

    with col_clear:
        clear_btn = st.button("Clear", use_container_width=True)


    if clear_btn:
        st.session_state.recording = None
        st.rerun() #reflect the cleared state immediately without waiting for another interaction

    # If the user clicked Record, start capturing audio from the microphone.
    if record_btn:
        # st.spinner shows a loading animation with a message while its block runs.
        with st.spinner("Recording... speak now!"):
            # sd.rec() starts recording from the default microphone.
            audio_data = sd.rec(
                int(RECORD_SECS * SAMPLE_RATE), #total number of samples to record (1* 16000)
                samplerate=SAMPLE_RATE, #how many samples per second
                channels=1, #mono audio (1 channel)
                dtype='float32' #store samples as 32-bit floats
            )
            # sd.wait() blocks (pauses execution) until the recording is finished.
            # Without this, the script would continue before the mic is done capturing.
            sd.wait()

        # sd.rec() returns shape (16000, 1) — 16000 rows, 1 column (mono channel).
        # .flatten() collapses it to 1D shape (16000,), which our MFCC function expects.
        # Store in session_state so it survives the next rerun when the user clicks Analyze.
        st.session_state.recording = audio_data.flatten()
        st.success("Recording complete!")

    # Only show the playback player and Analyze button if a recording exists.
    if st.session_state.recording is not None:
        # We create a BytesIO object (an in-memory file) so we never touch disk.
        buffer = io.BytesIO()
        # Multiplying by 32767 scales [-1.0, 1.0] → [-32767, 32767] (16-bit integer range).
        # .astype(np.int16) converts the array to 16-bit signed integers.
        wav.write(buffer, SAMPLE_RATE,
                  (st.session_state.recording * 32767).astype(np.int16))
        # .seek(0) rewinds it to the beginning so st.audio can read from the start
        buffer.seek(0)
        # Render an audio playback widget in the UI that plays the contents of our buffer.
        st.audio(buffer, format="audio/wav")

        if st.button("Analyze Recording", use_container_width=True):
            with st.spinner("Processing..."):
                result = recognize_from_array(st.session_state.recording, spk_model, cmd_model)
            # Pass the result dictionary to the display function
            show_result(result)


# Tab 2: Upload 
with tab2:
    st.markdown("**Upload a `.wav` file to analyze.**")
    st.info("Best results with 1-second clips at 16kHz , same format as training data.")

    # st.file_uploader renders a drag-and-drop / browse file input.
    # type=["wav"] restricts accepted files to .wav only.
    # Returns a file-like object when a file is uploaded, None if nothing uploaded yet.
    uploaded = st.file_uploader("Choose a WAV file", type=["wav"])

    # Only proceed if the user has actually uploaded something.
    if uploaded is not None:
        # Show a playback widget for the uploaded file directly.
        st.audio(uploaded, format="audio/wav")

        if st.button("Analyze File", use_container_width=True):
            # uploaded.read() reads the entire uploaded file as raw bytes.
            # io.BytesIO() wraps those bytes in a file-like object in memory.
            # This lets librosa.load() read it as if it were a file on disk —
            # librosa accepts both file paths and file-like objects.
            audio_bytes_io = io.BytesIO(uploaded.read())

            # librosa.load() decodes the WAV bytes into a float32 NumPy array.(amplitude values of the sound wave)
            # Returns (audio_array, actual_sample_rate)
            y_arr, _ = librosa.load(audio_bytes_io, sr=SAMPLE_RATE)

            with st.spinner("Processing..."):
                result = recognize_from_array(y_arr, spk_model, cmd_model)
            show_result(result)


# Footer 

st.markdown("---")
st.caption("Neural Network Based Speech Recognition")
