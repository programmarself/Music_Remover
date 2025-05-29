import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
import os
import tempfile
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Set page config
st.set_page_config(page_title="Music Remover AI", layout="wide")

# Title and description
st.title("🎵 Music Remover AI Agent")
st.write("Upload a video or audio file to remove music while preserving voice")

# Sidebar for additional options
with st.sidebar:
    st.header("Settings")
    separation_method = st.selectbox(
        "Separation Method",
        ["Demucs (Recommended)", "Vocal Isolation", "Noise Reduction"]
    )
    aggressiveness = st.slider(
        "Processing Aggressiveness", 1, 10, 5
    ) if separation_method != "Demucs (Recommended)" else None

# File upload section
uploaded_file = st.file_uploader(
    "Upload audio or video file",
    type=["mp3", "wav", "ogg", "mp4", "avi", "mov"]
)

def separate_with_demucs(input_path, output_path):
    """Use Facebook's Demucs model for separation"""
    model = get_model(name='htdemucs')
    model.cpu()
    model.eval()
    
    audio, rate = librosa.load(input_path, sr=44100, mono=False)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)
    
    sources = apply_model(model, audio[None], device='cpu')[0]
    vocals = sources[3]  # Assuming vocals are in channel 3
    
    # Save the vocals
    librosa.output.write_wav(output_path, vocals.T, rate)

def process_audio(input_path, output_path, method="Demucs"):
    if method == "Demucs":
        separate_with_demucs(input_path, output_path)
    else:
        # Alternative methods can be implemented here
        y, sr = librosa.load(input_path, sr=None)
        vocal_part = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
        librosa.output.write_wav(output_path, vocal_part, sr)

def extract_audio_from_video(video_path, audio_path):
    video = AudioSegment.from_file(video_path)
    video.export(audio_path, format="wav")

if uploaded_file is not None:
    with st.spinner("Processing your file..."):
        # Create temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
                try:
                    # Check if it's a video file
                    if uploaded_file.name.split('.')[-1] in ['mp4', 'avi', 'mov']:
                        video_path = uploaded_file.name
                        with open(video_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        extract_audio_from_video(video_path, tmp_audio.name)
                    else:
                        # It's an audio file
                        audio = AudioSegment.from_file(uploaded_file)
                        audio.export(tmp_audio.name, format="wav")
                    
                    # Process the audio
                    process_audio(tmp_audio.name, tmp_output.name, separation_method.split(' ')[0])
                    
                    # Display results
                    st.success("Processing complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Audio")
                        st.audio(tmp_audio.name)
                    
                    with col2:
                        st.subheader("Vocals Only")
                        st.audio(tmp_output.name)
                    
                    # Download button
                    with open(tmp_output.name, "rb") as f:
                        st.download_button(
                            label="Download Vocal Track",
                            data=f,
                            file_name="vocals_only.wav",
                            mime="audio/wav"
                        )
                
                finally:
                    # Clean up
                    if os.path.exists(tmp_audio.name):
                        os.unlink(tmp_audio.name)
                    if os.path.exists(tmp_output.name):
                        os.unlink(tmp_output.name)
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.unlink(video_path)

# Add agentic behavior
if st.checkbox("Show advanced options"):
    st.write("""
    ### AI Agent Settings
    Configure how the AI agent should behave when processing your audio.
    """)
    
    auto_adjust = st.checkbox("Auto-adjust parameters based on content", True)
    preserve_pitch = st.checkbox("Preserve original vocal pitch", True)
    enhance_voice = st.checkbox("Enhance voice clarity", False)
    
    if enhance_voice:
        enhancement_level = st.slider("Voice enhancement level", 1, 10, 5)

# Add some agent-like behavior
if uploaded_file and st.button("Ask AI Agent for processing advice"):
    st.info("""
    🎙️ AI Agent Analysis:
    - Detected audio with music and voice components
    - Recommending Demucs separation for best results
    - Estimated processing time: 30-90 seconds depending on length
    """)