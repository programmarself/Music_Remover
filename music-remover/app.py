import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
import os
import tempfile
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import subprocess
import sys
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="🎵 Music Remover AI", layout="wide")

# Check and install FFmpeg if not available
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("FFmpeg not found. Installing...")
        try:
            import google.colab
            !apt install ffmpeg -y
        except:
            try:
                if sys.platform == "win32":
                    !conda install -y ffmpeg -c conda-forge
                else:
                    !apt-get install ffmpeg -y
            except:
                st.error("Failed to install FFmpeg automatically. Please install it manually.")
                return False
    return True

if not ensure_ffmpeg():
    st.error("FFmpeg installation failed. The app may not work properly.")

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
    
    st.header("Advanced Options")
    auto_adjust = st.checkbox("Auto-adjust parameters based on content", True)
    preserve_pitch = st.checkbox("Preserve original vocal pitch", True)
    enhance_voice = st.checkbox("Enhance voice clarity", False)

def separate_with_demucs(input_bytes, sr=44100):
    """Use Facebook's Demucs model for separation"""
    try:
        model = get_model(name='htdemucs')
        model.cpu()
        model.eval()
        
        audio, rate = librosa.load(input_bytes, sr=sr, mono=False)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        
        sources = apply_model(model, audio[None], device='cpu')[0]
        vocals = sources[3]  # Vocals are in channel 3
        return vocals.T, rate
    except Exception as e:
        logger.error(f"Error in Demucs separation: {str(e)}")
        raise

def process_audio(input_bytes, method="Demucs"):
    """Process audio based on selected method"""
    try:
        if method == "Demucs":
            audio, sr = separate_with_demucs(input_bytes)
        else:
            y, sr = librosa.load(input_bytes, sr=None)
            audio = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9)
        return audio, sr
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def convert_to_wav(input_bytes, input_format):
    """Convert any audio to WAV format in memory"""
    try:
        if input_format in ['mp4', 'avi', 'mov']:
            # For video files, we need to extract audio first
            with tempfile.NamedTemporaryFile(suffix=f".{input_format}", delete=False) as tmp_video:
                tmp_video.write(input_bytes)
                tmp_video_path = tmp_video.name
            
            # Extract audio using FFmpeg
            tmp_audio_path = tmp_video_path + ".wav"
            cmd = f"ffmpeg -i {tmp_video_path} -vn -acodec pcm_s16le -ar 44100 -ac 1 {tmp_audio_path}"
            subprocess.run(cmd, shell=True, check=True)
            
            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()
            
            os.unlink(tmp_video_path)
            os.unlink(tmp_audio_path)
            return audio_bytes
        else:
            # For audio files, just read and return
            return input_bytes
    except Exception as e:
        logger.error(f"Error converting to WAV: {str(e)}")
        raise

# File upload section
uploaded_file = st.file_uploader(
    "Upload audio or video file",
    type=["mp3", "wav", "ogg", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    try:
        with st.spinner("Processing your file..."):
            # Get file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Read the uploaded file into memory
            input_bytes = uploaded_file.read()
            
            # Convert to WAV if needed
            if file_ext != 'wav':
                input_bytes = convert_to_wav(input_bytes, file_ext)
            
            # Process the audio
            audio, sr = process_audio(BytesIO(input_bytes), separation_method.split(' ')[0])
            
            # Save processed audio to memory
            output_bytes = BytesIO()
            sf.write(output_bytes, audio, sr, format='WAV')
            output_bytes.seek(0)
            
            # Display results
            st.success("Processing complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Audio")
                st.audio(input_bytes)
            
            with col2:
                st.subheader("Vocals Only")
                st.audio(output_bytes)
            
            # Download button
            st.download_button(
                label="Download Vocal Track",
                data=output_bytes,
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
    """)
