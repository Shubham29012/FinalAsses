import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import warnings
import io
import base64
import os
from datetime import datetime, timedelta
import bcrypt
import pandas as pd
import json

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
if 'username' not in st.session_state:
    st.session_state.username = ""

# Mock user database (in a real app, use a proper database)
if 'user_db' not in st.session_state:
    # Initialize with a default admin user (hash for "admin123")
    default_hash = bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()).decode()
    st.session_state.user_db = {
        'admin': {
            'password_hash': default_hash,
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
    }
    
    # Try to load existing users from file
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                st.session_state.user_db = json.load(f)
    except Exception as e:
        st.error(f"Error loading user database: {e}")

# Save user database function
def save_user_db():
    try:
        with open('users.json', 'w') as f:
            json.dump(st.session_state.user_db, f)
    except Exception as e:
        st.error(f"Error saving user database: {e}")

# App title and configuration
st.set_page_config(
    page_title="Speech Enhancement System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    # .login-box {
    #     padding: 2rem;
    #     background-color: #f0f2f6;
    #     border-radius: 10px;
    #     box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    #     max-width: 500px;
    #     margin: 0 auto;
    # }
    .stAudio {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Speech Enhancement Functions

# Improved spectral gating noise removal with proper handling of negative values
def remove_noise_spectral(y, sr, n_fft=256, hop=128, margin=3, power=2):
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude and phase
    S_full, phase = librosa.magphase(D)

    # Ensure all values are non-negative
    S_full = np.abs(S_full)

    # Estimate noise profile using non-local median filtering
    noise_frames = int(librosa.time_to_frames(2, sr=sr, hop_length=hop))
    S_filter = librosa.decompose.nn_filter(
        S_full,
        aggregate=np.median,
        metric='cosine',
        width=max(1, noise_frames)
    )

    # Ensure S_filter is non-negative
    S_filter = np.maximum(S_filter, 1e-10)

    # Calculate the noise component safely
    noise_component = np.maximum(S_full - S_filter, 0)

    # Apply soft mask - ensuring all inputs are non-negative
    try:
        mask = librosa.util.softmask(
            noise_component,
            margin * S_filter,
            power=power
        )
    except Exception as e:
        # Fall back to simple ratio mask
        mask = noise_component / (noise_component + margin * S_filter + 1e-10)
        mask = np.minimum(np.maximum(mask, 0), 1)  # Ensure mask is between 0 and 1

    # Apply mask to spectrogram
    S_foreground = S_full * (1 - mask)  # Keep the complement of the noise mask

    # Reconstruct signal with enhanced magnitude and original phase
    D_foreground = S_foreground * phase
    y_out = librosa.istft(D_foreground, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Spectral subtraction method with numerical stability
def spectral_subtraction(y, sr, n_fft=256, hop=128, alpha=2.0, beta=0.02):
    """
    Classic spectral subtraction for noise reduction.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude spectrogram
    mag_spec = np.abs(D)
    phase_spec = np.angle(D)

    # Estimate noise from the first few frames
    noise_estimate = np.mean(mag_spec[:, :max(5, int(mag_spec.shape[1] * 0.05))], axis=1)
    noise_estimate = np.maximum(noise_estimate, 1e-10).reshape(-1, 1)

    # Apply spectral subtraction with oversubtraction factor and flooring
    subtracted = np.maximum(mag_spec - alpha * noise_estimate, beta * mag_spec)

    # Reconstruct signal
    D_enhanced = subtracted * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Wiener filtering for speech enhancement with stability fixes
def wiener_filter(y, sr, n_fft=256, hop=128, noise_frames=5):
    """
    Wiener filtering for speech enhancement.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get power spectrogram
    power_spec = np.abs(D)**2

    # Estimate noise power from first few frames
    noise_power = np.mean(power_spec[:, :noise_frames], axis=1).reshape(-1, 1)
    noise_power = np.maximum(noise_power, 1e-10)

    # Compute Wiener filter with numerical safety
    speech_power = np.maximum(power_spec - noise_power, 0.01 * noise_power)
    wiener_gain = speech_power / (speech_power + noise_power + 1e-10)

    # Apply filter to complex spectrum
    D_enhanced = D * wiener_gain

    # Inverse STFT
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Speech presence probability based enhancement with stability
def speech_presence_enhance(y, sr, n_fft=256, hop=128, alpha=0.98, beta=0.8):
    """
    Enhancement using speech presence probability estimation.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude and phase
    mag_spec = np.abs(D)
    phase_spec = np.angle(D)

    # Initialize noise estimate (first frame)
    noise_est = np.maximum(mag_spec[:, 0], 1e-10).reshape(-1, 1)

    # Initialize enhanced spectrogram
    enhanced_spec = np.zeros_like(mag_spec)

    # Process each frame
    for i in range(mag_spec.shape[1]):
        # Current frame magnitude
        frame_mag = mag_spec[:, i].reshape(-1, 1)

        # Calculate a posteriori SNR with numerical safety
        post_snr = np.minimum((frame_mag**2) / (noise_est**2 + 1e-10), 1e6)

        # Estimate speech presence probability
        speech_prob = 1 / (1 + np.exp(-(post_snr - 1) * 2))

        # Update noise estimate (when speech probability is low)
        noise_est = alpha * noise_est + (1 - alpha) * frame_mag * (1 - speech_prob)
        noise_est = np.maximum(noise_est, 1e-10)

        # Calculate gain based on speech probability
        gain = beta * speech_prob + (1 - beta) * np.sqrt(speech_prob + 1e-10)

        # Apply gain
        enhanced_spec[:, i] = (frame_mag.flatten() * gain.flatten())

    # Reconstruct signal
    D_enhanced = enhanced_spec * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# MMSE-based speech enhancement with numerical stability
def mmse_stsa_enhance(y, sr, n_fft=256, hop=128, alpha=0.98, beta=0.8):
    """
    Minimum Mean Square Error Short-Time Spectral Amplitude enhancement.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude and phase
    mag_spec, phase_spec = np.abs(D), np.angle(D)

    # Initialize noise estimate from first few frames
    noise_psd = np.mean(mag_spec[:, :5]**2, axis=1).reshape(-1, 1)
    noise_psd = np.maximum(noise_psd, 1e-10)  # Ensure no zero values

    # Initialize enhanced spectrogram
    enhanced_spec = np.zeros_like(mag_spec)

    # Process each frame
    for i in range(mag_spec.shape[1]):
        # Current frame magnitude
        frame_mag = mag_spec[:, i].reshape(-1, 1)

        # Calculate a posteriori SNR with numerical safety
        post_snr = np.minimum((frame_mag**2) / (noise_psd + 1e-10), 1e6)

        # Decision-directed approach for a priori SNR estimation
        if i > 0:
            prior_snr_term = (enhanced_spec[:, i-1].reshape(-1, 1)**2) / (noise_psd + 1e-10)
            prior_snr_term = np.minimum(prior_snr_term, 1e6)  # Prevent overflow
            prior_snr = alpha * prior_snr_term + (1-alpha) * np.maximum(post_snr - 1, 0)
        else:
            prior_snr = np.maximum(post_snr - 1, 0)

        # Calculate v with numerical safety
        v = np.minimum(prior_snr * post_snr / (1 + prior_snr + 1e-10), 100)

        # Compute MMSE gain with numerical safety
        sqrt_v = np.sqrt(v + 1e-10)
        exp_term = np.minimum(np.exp(v/2), 1e10)
        gain = (sqrt_v / (1 + v + 1e-10)) * exp_term

        # Apply gain
        enhanced_spec[:, i] = (frame_mag.flatten() * gain.flatten())

        # Update noise estimate periodically for stability
        if i % 10 == 0:
            noise_psd = 0.99 * noise_psd + 0.01 * (frame_mag**2)
            noise_psd = np.maximum(noise_psd, 1e-10)

    # Reconstruct signal
    D_enhanced = enhanced_spec * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is finite and valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Voice activity detection for better noise estimation
def vad_enhance(y, sr, n_fft=256, hop=128, threshold=1.5):
    """
    Enhancement with voice activity detection for better noise estimation.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude
    mag_spec = np.abs(D)
    phase_spec = np.angle(D)

    # Calculate frame energy
    frame_energy = np.sum(mag_spec**2, axis=0)

    # Normalize energy
    norm_energy = frame_energy / (np.mean(frame_energy) + 1e-10)

    # Detect speech frames (VAD)
    speech_frames = norm_energy > threshold

    # Get noise-only frames
    noise_frames = ~speech_frames

    # If no noise frames, assume first 5% of frames are noise
    if not np.any(noise_frames):
        noise_frames[:int(len(noise_frames) * 0.05)] = True

    # Estimate noise spectrum
    noise_spec = np.mean(mag_spec[:, noise_frames], axis=1).reshape(-1, 1)
    noise_spec = np.maximum(noise_spec, 1e-10)

    # Apply Wiener filter with improved noise estimate
    speech_power = np.maximum(mag_spec**2 - noise_spec**2, 0.01 * noise_spec**2)
    wiener_gain = speech_power / (speech_power + noise_spec**2 + 1e-10)

    # Apply filter
    enhanced_spec = mag_spec * np.sqrt(wiener_gain + 1e-10)

    # Reconstruct signal
    D_enhanced = enhanced_spec * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Formant enhancement for speech clarity
def formant_enhance(y, sr, n_fft=256, hop=128, formant_boost=3.0):
    """
    Enhance speech formants to improve clarity.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude and phase
    mag_spec = np.abs(D)
    phase_spec = np.angle(D)

    # Typical formant frequency ranges (in Hz)
    formant_ranges = [
        (300, 1000),   # F1: vowel recognition
        (1000, 2500),  # F2: front/back vowel distinction
        (2500, 4000)   # F3: additional clarity
    ]

    # Create formant boost filter
    formant_filter = np.ones(n_fft//2 + 1)

    # Apply boosts to formant ranges
    for f_min, f_max in formant_ranges:
        # Convert Hz to bins
        bin_min = max(0, int(f_min * n_fft / sr))
        bin_max = min(n_fft//2, int(f_max * n_fft / sr))

        # Apply boost to formant region
        formant_filter[bin_min:bin_max] = formant_boost

    # Apply filter to magnitude spectrogram
    enhanced_spec = mag_spec * formant_filter.reshape(-1, 1)

    # Reconstruct signal
    D_enhanced = enhanced_spec * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Normalize and ensure valid output
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-10) * np.max(np.abs(y))
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Adaptive equalizer for speech enhancement
def adaptive_equalizer(y, sr, n_fft=256, hop=128, n_bands=6, adapt_rate=0.05):
    """
    Apply adaptive equalization to enhance speech.
    """
    # Short-time Fourier transform
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    # Get magnitude and phase
    mag_spec = np.abs(D)
    phase_spec = np.angle(D)

    # Calculate total number of frequency bins
    n_bins = n_fft//2 + 1

    # Create frequency bands (mel-like spacing)
    bands = np.logspace(np.log10(20), np.log10(sr/2), n_bands+1)
    band_bins = np.round(bands * n_bins / (sr/2)).astype(int)
    band_bins = np.minimum(band_bins, n_bins)

    # Initialize equalizer gains
    eq_gains = np.ones(n_bands)

    # Target spectrum shape (approximating long-term speech spectrum)
    target_shape = np.array([0.5, 1.0, 1.5, 1.2, 0.8, 0.5])

    # Create equalizer filter
    eq_filter = np.ones(n_bins)

    # Calculate average spectrum
    avg_spec = np.mean(mag_spec, axis=1)

    # Calculate band energies
    band_energies = np.zeros(n_bands)
    for b in range(n_bands):
        start, end = band_bins[b], band_bins[b+1]
        band_energies[b] = np.mean(avg_spec[start:end])

    # Normalize band energies
    if np.sum(band_energies) > 0:
        band_energies = band_energies / (np.mean(band_energies) + 1e-10)

    # Calculate adaptive gains
    for b in range(n_bands):
        if band_energies[b] > 0:
            eq_gains[b] = target_shape[b] / (band_energies[b] + 1e-10)

            # Limit gain range for stability
            eq_gains[b] = np.clip(eq_gains[b], 0.2, 5.0)

    # Apply smoothing to gains
    eq_gains = 1.0 + adapt_rate * (eq_gains - 1.0)

    # Build equalizer filter
    for b in range(n_bands):
        start, end = band_bins[b], band_bins[b+1]
        eq_filter[start:end] = eq_gains[b]

    # Apply filter smoothly across spectrum (avoid sharp transitions)
    eq_filter = signal.savgol_filter(eq_filter, 11, 2)

    # Apply equalizer to magnitude spectrogram
    enhanced_spec = mag_spec * eq_filter.reshape(-1, 1)

    # Reconstruct signal
    D_enhanced = enhanced_spec * np.exp(1j * phase_spec)
    y_out = librosa.istft(D_enhanced, hop_length=hop, length=len(y))

    # Ensure output is valid
    y_out = np.nan_to_num(y_out)
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out

# Combined method with error handling
def combined_speech_enhancement(y, sr, method="balanced"):
    """
    Combined speech enhancement approach using multiple techniques.
    """
    n_fft, hop = 512, 128  # Larger FFT for better frequency resolution

    try:
        if method == "noise_reduction":
            # Focus on noise reduction
            st.info("Applying noise reduction pipeline...")
            y = wiener_filter(y, sr, n_fft, hop)
            y = vad_enhance(y, sr, n_fft, hop)

        elif method == "clarity":
            # Focus on speech clarity
            st.info("Applying speech clarity pipeline...")
            try:
                y = mmse_stsa_enhance(y, sr, n_fft, hop)
            except Exception as e:
                st.warning(f"MMSE enhancement failed, falling back to spectral subtraction: {e}")
                y = spectral_subtraction(y, sr, n_fft, hop)
            y = formant_enhance(y, sr, n_fft, hop)

        else:  # balanced approach
            st.info("Applying balanced enhancement pipeline...")
            # First remove noise
            y = remove_noise_spectral(y, sr, n_fft, hop)

            # Then enhance speech characteristics
            y = speech_presence_enhance(y, sr, n_fft, hop)

            # Finally, improve clarity
            y = adaptive_equalizer(y, sr, n_fft, hop)

        # Ensure final output is valid
        y = np.nan_to_num(y)
        y = np.clip(y, -1.0, 1.0)
        return y

    except Exception as e:
        st.error(f"Error in enhancement pipeline: {e}")
        # Return original audio if processing fails
        return y

# Plot audio analysis
def plot_analysis(orig, proc, sr, n_fft=512, hop=128, title="Audio Enhancement"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot waveforms
    axs[0, 0].set_title("Original Waveform")
    axs[0, 0].plot(orig[:min(8000, len(orig))])
    axs[0, 0].set_xlabel("Samples")
    axs[0, 0].set_ylabel("Amplitude")

    axs[0, 1].set_title("Enhanced Waveform")
    axs[0, 1].plot(proc[:min(8000, len(proc))])
    axs[0, 1].set_xlabel("Samples")
    axs[0, 1].set_ylabel("Amplitude")

    # Plot spectrograms
    axs[1, 0].set_title("Original Spectrogram")
    spec0 = axs[1, 0].specgram(orig, NFFT=n_fft, Fs=sr, noverlap=n_fft-hop, cmap='viridis')
    fig.colorbar(spec0[3], ax=axs[1, 0], format='%+2.0f dB')
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Frequency (Hz)")

    axs[1, 1].set_title("Enhanced Spectrogram")
    spec1 = axs[1, 1].specgram(proc, NFFT=n_fft, Fs=sr, noverlap=n_fft-hop, cmap='viridis')
    fig.colorbar(spec1[3], ax=axs[1, 1], format='%+2.0f dB')
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Frequency (Hz)")

    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

# Signal quality metrics
def compute_metrics(orig, proc):
    """
    Compute various audio quality metrics.
    """
    # Ensure same length
    min_len = min(len(orig), len(proc))
    orig, proc = orig[:min_len], proc[:min_len]

    # Signal-to-Noise Ratio (assuming proc is enhanced version)
    def snr(clean, noisy):
        noise = clean - noisy
        return 10 * np.log10((np.sum(clean**2) + 1e-10) / (np.sum(noise**2) + 1e-10))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((orig - proc)**2))

    # Peak-to-Average Power Ratio
    def papr(signal):
        return 10 * np.log10((np.max(signal**2) + 1e-10) / (np.mean(signal**2) + 1e-10))

    # Perceptual metrics approximation
    # (These are rough approximations of perceptual metrics)

    # Approximate clarity - higher frequency content ratio
    def approx_clarity(signal, sr):
        D = librosa.stft(signal)
        mag = np.abs(D)
        # Ratio of high freq to total energy
        high_freq_cutoff = int(2000 * len(mag) / (sr/2))  # 2kHz cutoff
        high_energy = np.sum(mag[high_freq_cutoff:])
        total_energy = np.sum(mag)
        return high_energy / (total_energy + 1e-10)

    metrics = {
        'SNR (dB)': snr(proc, orig),  # Higher is better
        'RMSE': rmse,  # Lower is better
        'PAPR Orig (dB)': papr(orig),
        'PAPR Proc (dB)': papr(proc),
        'Clarity Orig': approx_clarity(orig, 16000),  # Using default sr=16k
        'Clarity Proc': approx_clarity(proc, 16000)
    }

    return metrics

# Function to get file download link
def get_audio_download_link(audio_data, sr, filename, text):
    """Generate a download link for audio file"""
    buffer = io.BytesIO()
    torchaudio.save(buffer, torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0), sr)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Login form
def show_login_form():
    st.markdown("<h1 class='main-header'>Speech Enhancement System</h1>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Login", use_container_width=True):
                    # Check if username exists
                    if username in st.session_state.user_db:
                        stored_hash = st.session_state.user_db[username]['password_hash']
                        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            # Update last login time
                            st.session_state.user_db[username]['last_login'] = datetime.now().isoformat()
                            save_user_db()
                            st.rerun()
                        else:
                            st.error("Invalid password")
                    else:
                        st.error("Username not found")
            
            with col2:
                if st.button("Demo Login", use_container_width=True):
                    st.session_state.logged_in = True
                    st.session_state.username = "guest"
                    st.rerun()
        
        with register_tab:
            new_username = st.text_input("Choose Username", key="reg_username")
            new_password = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Register", use_container_width=True):
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif new_username in st.session_state.user_db:
                        st.error("Username already exists")
                    else:
                        # Hash password and store new user
                        hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                        st.session_state.user_db[new_username] = {
                            'password_hash': hashed_pw,
                            'created_at': datetime.now().isoformat(),
                            'last_login': None
                        }
                        save_user_db()
                        st.success("Registration successful! You can now log in.")
                else:
                    st.warning("Username and password are required")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Main application page
def show_app():
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/microphone.png", width=100)
        st.title(f"Welcome, {st.session_state.username}!")
        
        page = st.radio("Navigation", 
                        ["Dashboard", 
                         "Enhance Audio", 
                         "Advanced Options",
                         "User Profile"])
        
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    # Main page content based on selection
    if page == "Dashboard":
        show_dashboard()
    elif page == "Enhance Audio":
        show_enhance_audio()
    elif page == "Advanced Options":
        show_advanced_options()
    elif page == "User Profile":
        show_user_profile()

# Dashboard page
def show_dashboard():
    st.markdown("<h1 class='main-header'>Speech Enhancement Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>About</div>", unsafe_allow_html=True)
        st.write("""
        This application provides various speech enhancement techniques to improve audio quality. 
        You can use it to:
        
        - Remove background noise from audio recordings
        - Enhance speech clarity and intelligibility
        - Improve the overall audio quality
        - Compare different enhancement techniques
        """)
        
        st.markdown("<div class='sub-header'>How to Use</div>", unsafe_allow_html=True)
        st.write("""
        1. Navigate to the **Enhance Audio** section for standard enhancement
        2. Upload your audio file (WAV, MP3, FLAC formats supported)
        3. Choose an enhancement method
        4. Process your audio and download the enhanced version
        5. For more control, visit the **Advanced Options** section
        """)

    with col2:
        st.markdown("<div class='sub-header'>Features</div>", unsafe_allow_html=True)
        st.info("✅ Noise Reduction: Remove background noise and hums")
        st.info("✅ Speech Clarity: Enhance voice intelligibility")
        st.info("✅ Advanced Options: Fine-tune specific enhancement techniques")
        st.info("✅ Audio Analysis: Visualize before and after spectrograms")
        st.info("✅ Quality Metrics: Evaluate enhancement effectiveness")
    
    st.markdown("<div class='sub-header'>Enhancement Techniques Overview</div>", unsafe_allow_html=True)
    
    tech_df = pd.DataFrame({
        'Technique': [
            'Spectral Gating', 
            'Wiener Filtering', 
            'MMSE STSA', 
            'VAD-Based Enhancement',
            'Formant Enhancement',
            'Adaptive Equalization'
        ],
        'Best For': [
            'General noise reduction',
            'Stationary background noise',
            'Speech in heavy noise',
            'Speech with pauses',
            'Voice clarity improvement',
            'Frequency balance adjustment'
        ],
        'Description': [
            'Removes noise by filtering in frequency domain',
            'Statistical filter that minimizes noise',
            'Minimum Mean Square Error approach for speech enhancement',
            'Uses voice activity detection for better noise estimation',
            'Enhances speech formants for better intelligibility',
            'Adjusts frequency bands adaptively for optimal sound'
        ]
    })
    
    st.dataframe(tech_df, use_container_width=True)

# Enhance Audio page
def show_enhance_audio():
    st.markdown("<h1 class='main-header'>Enhance Audio</h1>", unsafe_allow_html=True)
    
    # Audio file upload
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load audio file
        try:
            y, sr = librosa.load("temp_audio.wav", sr=None)
            st.success(f"File loaded: {uploaded_file.name}, Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f}s")
            
            # Display original audio player
            st.audio("temp_audio.wav")
            
            # Enhancement options
            st.markdown("<div class='sub-header'>Enhancement Options</div>", unsafe_allow_html=True)
            
            method = st.radio(
                "Select enhancement method:",
                ["Noise Reduction Focus", "Speech Clarity Focus"]
            )
            
            # Map selection to method
            method_map = {
                "Balanced Enhancement": "balanced",
                "Noise Reduction Focus": "noise_reduction",
                "Speech Clarity Focus": "clarity"
            }
            
            # Process button
            if st.button("Enhance Audio", type="primary", use_container_width=True):
                with st.spinner("Processing audio..."):
                    # Process the audio
                    enhanced = combined_speech_enhancement(y, sr, method_map[method])
                    
                    # Save enhanced audio
                    out_filename = f"enhanced_{uploaded_file.name}"
                    torchaudio.save(out_filename, torch.from_numpy(enhanced.astype(np.float32)).unsqueeze(0), sr)
                    
                    # Play enhanced audio
                    st.markdown("<div class='sub-header'>Enhanced Audio</div>", unsafe_allow_html=True)
                    st.audio(out_filename)
                    
                    # Download link
                    st.markdown(get_audio_download_link(enhanced, sr, out_filename, "Download Enhanced Audio"), unsafe_allow_html=True)
                    
                    # Show analysis
                    st.markdown("<div class='sub-header'>Audio Analysis</div>", unsafe_allow_html=True)
                    fig = plot_analysis(y, enhanced, sr, title=f"Speech Enhancement ({method})")
                    st.pyplot(fig)
                    
                    # Show metrics
                    st.markdown("<div class='sub-header'>Enhancement Metrics</div>", unsafe_allow_html=True)
                    metrics = compute_metrics(y, enhanced)
                    
                    # Format metrics for display
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Interpretation of metrics
                    st.info("""
                    **Metrics Interpretation:**
                    - **SNR (dB)**: Signal-to-Noise Ratio - Higher is better
                    - **RMSE**: Root Mean Square Error - Lower is better
                    - **PAPR**: Peak-to-Average Power Ratio - Measures dynamic range
                    - **Clarity**: Higher frequency content ratio - Higher suggests better intelligibility
                    """)
        
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
    
    else:
        # Show sample audio option
        st.markdown("### No file uploaded")
        st.info("Upload an audio file to get started, or use a sample file.")
        
        if st.button("Use Sample Audio"):
            # Generate sample audio (simple sine wave with noise)
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Generate a speech-like signal (multiple frequencies)
            signal = 0.5 * np.sin(2 * np.pi * 220 * t)  # Fundamental
            signal += 0.3 * np.sin(2 * np.pi * 440 * t)  # First harmonic
            signal += 0.1 * np.sin(2 * np.pi * 880 * t)  # Second harmonic
            
            # Add white noise
            noise = np.random.normal(0, 0.1, len(signal))
            noisy_signal = signal + noise
            
            # Save the sample file
            torchaudio.save("sample_audio.wav", 
                            torch.from_numpy(noisy_signal.astype(np.float32)).unsqueeze(0), 
                            sample_rate)
            
            st.audio("sample_audio.wav")
            st.success("Sample noisy audio loaded. You can now proceed with enhancement.")
            
            # Reload the page to show options with the sample audio
            st.experimental_rerun()

# Advanced Options page
def show_advanced_options():
    st.markdown("<h1 class='main-header'>Advanced Enhancement Options</h1>", unsafe_allow_html=True)
    
    st.write("""
    This section allows you to apply individual enhancement techniques and fine-tune their parameters.
    Upload your audio file and select from the available techniques below.
    """)
    
    # Audio file upload
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load audio file
        try:
            y, sr = librosa.load("temp_audio.wav", sr=None)
            st.success(f"File loaded: {uploaded_file.name}, Sample rate: {sr} Hz, Duration: {len(y)/sr:.2f}s")
            
            # Display original audio player
            st.audio("temp_audio.wav")
            
            # Enhancement technique selection
            st.markdown("<div class='sub-header'>Select Enhancement Technique</div>", unsafe_allow_html=True)
            
            technique = st.selectbox(
                "Enhancement technique:",
                [
                    "Spectral Gating (Noise Removal)",
                    "Wiener Filtering",
                    "MMSE STSA Enhancement",
                    "VAD-Based Enhancement",
                    "Formant Enhancement",
                    "Adaptive Equalization"
                ]
            )
            
            # Parameters based on selected technique
            st.markdown("<div class='sub-header'>Technique Parameters</div>", unsafe_allow_html=True)
            
            n_fft = st.slider("FFT Size", min_value=128, max_value=2048, value=512, step=128,
                              help="Size of the Fast Fourier Transform (FFT) window. Larger values give better frequency resolution but worse time resolution.")
            
            hop = st.slider("Hop Length", min_value=64, max_value=512, value=128, step=64,
                           help="Number of samples between successive frames. Smaller values give more overlap but increase computation time.")
            
            # Additional parameters based on technique
            if technique == "Spectral Gating (Noise Removal)":
                margin = st.slider("Margin", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                                  help="Controls sensitivity of the noise gate. Higher values preserve more of the original signal.")
                power = st.slider("Power", min_value=1, max_value=4, value=2, step=1,
                                 help="Exponent for spectral mask. Higher values create sharper transitions.")
                
            elif technique == "Wiener Filtering":
                noise_frames = st.slider("Noise Frames", min_value=1, max_value=20, value=5, step=1,
                                        help="Number of initial frames to use for noise estimation.")
                
            elif technique == "MMSE STSA Enhancement":
                alpha = st.slider("Alpha", min_value=0.8, max_value=0.99, value=0.98, step=0.01,
                                 help="Controls adaptation speed. Higher values put more weight on previous frames.")
                beta = st.slider("Beta", min_value=0.5, max_value=0.95, value=0.8, step=0.05,
                                help="Controls smoothing of gain application.")
                
            elif technique == "VAD-Based Enhancement":
                threshold = st.slider("VAD Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                                     help="Energy threshold for voice activity detection. Higher values detect fewer speech frames.")
                
            elif technique == "Formant Enhancement":
                formant_boost = st.slider("Formant Boost", min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                                         help="Gain applied to formant frequency ranges. Higher values increase speech clarity but may sound unnatural.")
                
            elif technique == "Adaptive Equalization":
                n_bands = st.slider("Number of Bands", min_value=3, max_value=12, value=6, step=1,
                                   help="Number of frequency bands for equalization.")
                adapt_rate = st.slider("Adaptation Rate", min_value=0.01, max_value=0.2, value=0.05, step=0.01,
                                      help="Controls how quickly the equalizer adapts. Higher values are more aggressive.")
            
            # Process button
            if st.button("Apply Enhancement", type="primary", use_container_width=True):
                with st.spinner("Processing audio..."):
                    # Apply the selected technique with parameters
                    if technique == "Spectral Gating (Noise Removal)":
                        enhanced = remove_noise_spectral(y, sr, n_fft, hop, margin, power)
                        method_name = "spectral_gating"
                    elif technique == "Wiener Filtering":
                        enhanced = wiener_filter(y, sr, n_fft, hop, noise_frames)
                        method_name = "wiener_filter"
                    elif technique == "MMSE STSA Enhancement":
                        enhanced = mmse_stsa_enhance(y, sr, n_fft, hop, alpha, beta)
                        method_name = "mmse_stsa"
                    elif technique == "VAD-Based Enhancement":
                        enhanced = vad_enhance(y, sr, n_fft, hop, threshold)
                        method_name = "vad_enhance"
                    elif technique == "Formant Enhancement":
                        enhanced = formant_enhance(y, sr, n_fft, hop, formant_boost)
                        method_name = "formant_enhance"
                    elif technique == "Adaptive Equalization":
                        enhanced = adaptive_equalizer(y, sr, n_fft, hop, n_bands, adapt_rate)
                        method_name = "adaptive_eq"
                    
                    # Save enhanced audio
                    out_filename = f"{method_name}_{uploaded_file.name}"
                    torchaudio.save(out_filename, torch.from_numpy(enhanced.astype(np.float32)).unsqueeze(0), sr)
                    
                    # Play enhanced audio
                    st.markdown("<div class='sub-header'>Enhanced Audio</div>", unsafe_allow_html=True)
                    st.audio(out_filename)
                    
                    # Download link
                    st.markdown(get_audio_download_link(enhanced, sr, out_filename, "Download Enhanced Audio"), unsafe_allow_html=True)
                    
                    # Show analysis
                    st.markdown("<div class='sub-header'>Audio Analysis</div>", unsafe_allow_html=True)
                    fig = plot_analysis(y, enhanced, sr, title=f"{technique}")
                    st.pyplot(fig)
                    
                    # Show metrics
                    st.markdown("<div class='sub-header'>Enhancement Metrics</div>", unsafe_allow_html=True)
                    metrics = compute_metrics(y, enhanced)
                    
                    # Format metrics for display
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")

# User Profile page
def show_user_profile():
    st.markdown("<h1 class='main-header'>User Profile</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display user avatar
        st.image("https://img.icons8.com/fluency/240/000000/user-male-circle.png", width=150)
        st.markdown(f"<h2 style='text-align: center;'>{st.session_state.username}</h2>", unsafe_allow_html=True)
        
        # Show user info if not guest
        if st.session_state.username != "guest" and st.session_state.username in st.session_state.user_db:
            user_data = st.session_state.user_db[st.session_state.username]
            created_date = datetime.fromisoformat(user_data['created_at']).strftime("%Y-%m-%d")
            
            st.markdown(f"**Account created:** {created_date}")
            
            if user_data['last_login']:
                last_login = datetime.fromisoformat(user_data['last_login']).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"**Last login:** {last_login}")
    
    with col2:
        # Settings section
        st.markdown("<div class='sub-header'>Account Settings</div>", unsafe_allow_html=True)
        
        if st.session_state.username == "guest":
            st.warning("You are using a guest account. Create a real account to save your preferences.")
        else:
            # Change password option
            with st.expander("Change Password"):
                current_password = st.text_input("Current Password", type="password", key="current_pw")
                new_password = st.text_input("New Password", type="password", key="new_pw")
                confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pw")
                
                if st.button("Update Password"):
                    if st.session_state.username in st.session_state.user_db:
                        stored_hash = st.session_state.user_db[st.session_state.username]['password_hash']
                        if bcrypt.checkpw(current_password.encode(), stored_hash.encode()):
                            if new_password == confirm_password:
                                # Update password
                                hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                                st.session_state.user_db[st.session_state.username]['password_hash'] = hashed_pw
                                save_user_db()
                                st.success("Password updated successfully!")
                            else:
                                st.error("New passwords do not match")
                        else:
                            st.error("Current password is incorrect")
        
        # Enhancement preferences
        st.markdown("<div class='sub-header'>Enhancement Preferences</div>", unsafe_allow_html=True)
        
        default_method = st.selectbox(
            "Default enhancement method",
            ["Balanced Enhancement", "Noise Reduction Focus", "Speech Clarity Focus"]
        )
        
        default_fft = st.slider("Default FFT size", 128, 2048, 512, 128)
        default_hop = st.slider("Default hop length", 64, 512, 128, 64)
        
        if st.button("Save Preferences", type="primary"):
            st.success("Preferences saved!")
            # In a real app, these would be saved to the user's profile

        # Usage statistics
        st.markdown("<div class='sub-header'>Usage Statistics</div>", unsafe_allow_html=True)
        
        # Mock data for demo purposes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", "0")
        with col2:
            st.metric("Storage Used", "0 MB")
        with col3:
            st.metric("Processing Time", "0 min")

# Run the app
def main():
    # Check login state
    if not st.session_state.logged_in:
        show_login_form()
    else:
        show_app()

if __name__ == "__main__":
    main()