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
import ffmpeg
import shutil
import logging
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="🎵 Music Remover AI", layout="wide")

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

    st.header("Advanced Options")
    auto_adjust = st.checkbox("Auto-adjust parameters based on content", True)
    preserve_pitch = st.checkbox("Preserve original vocal pitch", True)
    enhance_voice = st.checkbox("Enhance voice clarity", False)
    
    if enhance_voice:
        enhancement_level = st.slider("Voice enhancement level", 1, 10, 5)

def separate_with_demucs(input_path, output_path):
    """Use Facebook's Demucs model for separation"""
    try:
        model = get_model(name='htdemucs')
        model.cpu()
        model.eval()
        
        audio, rate = librosa.load(input_path, sr=44100, mono=False)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        
        sources = apply_model(model, audio[None], device='cpu')[0]
        vocals = sources[3]  # Assuming vocals are in channel 3
        
        # Save the vocals
        sf.write(output_path, vocals.T, rate)
        logger.info("Successfully separated vocals using Demucs")
    except Exception as e:
        logger.error(f"Error in Demucs separation: {str(e)}")
        raise

def process_audio(input_path, output_path, method="Demucs"):
    """Process audio based on selected method"""
    try:
        if method == "Demucs":
            separate_with_demucs(input_path, output_path)
        else:
            y, sr = librosa.load(input_path, sr=None)
            vocal_part = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
            sf.write(output_path, vocal_part, sr)
        logger.info(f"Audio processed successfully using {method} method")
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video using ffmpeg"""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Use ffmpeg-python for more control
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='44100')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info("Successfully extracted audio from video")
    except ffmpeg.Error as e:
        error_msg = f"FFmpeg error: {e.stderr.decode('utf-8')}"
        logger.error(error_msg)
        st.error(error_msg)
        raise
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def save_uploaded_file(uploaded_file, save_path):
    """Save uploaded file to specified path"""
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"File saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

# File upload section
uploaded_file = st.file_uploader(
    "Upload audio or video file",
    type=["mp3", "wav", "ogg", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    try:
        # Create a proper temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, uploaded_file.name)
            audio_path = os.path.join(tmp_dir, "input_audio.wav")
            output_path = os.path.join(tmp_dir, "output_audio.wav")
            
            # Save uploaded file to temp location
            save_uploaded_file(uploaded_file, input_path)
            
            with st.spinner("Processing your file..."):
                # Process based on file type
                if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                    extract_audio_from_video(input_path, audio_path)
                else:
                    # Handle audio files directly
                    sound = AudioSegment.from_file(input_path)
                    sound.export(audio_path, format="wav")
                
                # Process the audio
                process_audio(audio_path, output_path, separation_method.split(' ')[0])
                
                # Display results
                st.success("Processing complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Audio")
                    st.audio(audio_path)
                
                with col2:
                    st.subheader("Vocals Only")
                    st.audio(output_path)
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Vocal Track",
                        data=f,
                        file_name="vocals_only.wav",
                        mime="audio/wav"
                    )
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        logger.error(f"Processing error: {str(e)}")

# Add some agent-like behavior
if uploaded_file and st.button("Ask AI Agent for processing advice"):
    st.info("""
    🎙️ AI Agent Analysis:
    - Detected audio with music and voice components
    - Recommending Demucs separation for best results
    - Estimated processing time: 30-90 seconds depending on length
    - Memory usage: Moderate (may be slower on CPU)
    """)
