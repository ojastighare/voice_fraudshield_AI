import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Directories
SAMPLE_DIR = "data/sample_audio"
VOICEPRINT_DIR = "data/enrolled_voiceprints"

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(VOICEPRINT_DIR, exist_ok=True)

SAMPLE_RATE = 16000

def generate_natural_human_voice(duration_sec=3.5, base_f0=125.0, speaker_name="Rajesh Sharma (CEO)"):
    """
    Generates a realistic synthetic representation of natural human speech
    with natural F0 pitch inflections, vocal fold micro-jitter, shimmer, and formant harmonics.
    """
    num_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Natural pitch contour (speech melody with continuous gentle variation)
    f0_contour = base_f0 + 15.0 * np.sin(2 * np.pi * 0.8 * t) + 8.0 * np.cos(2 * np.pi * 1.7 * t)

    # Micro-jitter (random tiny frequency fluctuation)
    jitter = np.random.normal(0, 0.8, num_samples)
    f0_contour += jitter

    # Phase integration for natural pitch
    phase = 2 * np.pi * np.cumsum(f0_contour) / SAMPLE_RATE

    # Formant harmonics (natural vocal tract resonances: F1=500Hz, F2=1500Hz, F3=2500Hz)
    h1 = np.sin(phase)
    h2 = 0.5 * np.sin(2 * phase)
    h3 = 0.25 * np.sin(3 * phase)
    h4 = 0.12 * np.sin(4 * phase)
    raw_speech = h1 + h2 + h3 + h4

    # Micro-shimmer (amplitude envelope variation + speech pauses)
    amplitude_env = 0.6 + 0.35 * np.sin(2 * np.pi * 2.5 * t) + 0.1 * np.cos(2 * np.pi * 5.0 * t)
    # Add natural speech pause between words
    pause_mask = (t < 1.0) | ((t > 1.2) & (t < 2.3)) | (t > 2.5)
    amplitude_env = amplitude_env * pause_mask

    natural_audio = raw_speech * amplitude_env

    # Soft low-pass filtering (natural vocal tract attenuation at high frequencies)
    b, a = scipy.signal.butter(4, 3800 / (SAMPLE_RATE / 2), btype='low')
    natural_audio = scipy.signal.filtfilt(b, a, natural_audio)

    # Normalize to 16-bit PCM range
    natural_audio = natural_audio / np.max(np.abs(natural_audio) + 1e-5) * 0.9
    pcm_audio = (natural_audio * 32767).astype(np.int16)

    return pcm_audio, natural_audio

def generate_ai_voice_clone(duration_sec=3.5, base_f0=125.0, vocoder_type="ElevenLabs Neural Vocoder"):
    """
    Generates a representation of an AI-cloned voice / neural TTS output.
    Features: Unnaturally flat F0, lack of natural vocal fold jitter, high-frequency vocoder phase artifacts,
    and unnatural sudden pitch jumps.
    """
    num_samples = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Synthetic F0: Unnaturally uniform pitch (robotic stability) with artificial pitch jump at t=1.8s
    f0_contour = np.full(num_samples, base_f0)
    f0_contour[t > 1.8] += 48.0 # Unnatural abrupt pitch spike (common in neural TTS synthesis)

    phase = 2 * np.pi * np.cumsum(f0_contour) / SAMPLE_RATE

    # Neural vocoder buzz (harsh phase alignment)
    h1 = np.sin(phase)
    h2 = 0.7 * np.sin(2 * phase + 0.5)
    h3 = 0.5 * np.sin(3 * phase + 1.0)
    h4 = 0.4 * np.sin(4 * phase + 1.5)
    h5 = 0.3 * np.sin(5 * phase + 2.0)
    # Vocoder high-frequency metallic artifact (> 6.8 kHz)
    hf_noise = 0.25 * np.sin(2 * np.pi * 7200 * t)

    synthetic_speech = h1 + h2 + h3 + h4 + h5 + hf_noise

    # Flat artificial amplitude envelope (too smooth, zero natural shimmer)
    amplitude_env = 0.8 * np.ones(num_samples)
    amplitude_env[(t > 1.5) & (t < 1.6)] = 0.1 # Mechanical pause

    synthetic_audio = synthetic_speech * amplitude_env

    # Normalize
    synthetic_audio = synthetic_audio / np.max(np.abs(synthetic_audio) + 1e-5) * 0.9
    pcm_audio = (synthetic_audio * 32767).astype(np.int16)

    return pcm_audio, synthetic_audio

def build_dataset():
    print("Generating synthetic and genuine audio sample suite...")

    samples_metadata = []

    # Sample 1: Genuine CEO Speech (English - Indian Accent)
    pcm1, float1 = generate_natural_human_voice(duration_sec=3.5, base_f0=120.0, speaker_name="Rajesh Sharma (CEO)")
    path1 = os.path.join(SAMPLE_DIR, "sample_genuine_ceo_english.wav")
    wav.write(path1, SAMPLE_RATE, pcm1)
    samples_metadata.append({
        "id": "sample_genuine_ceo_english",
        "title": "Rajesh Sharma (CEO) - Genuine Call",
        "language": "Indian English",
        "type": "GENUINE",
        "speaker": "Rajesh Sharma (CEO)",
        "file_path": "data/sample_audio/sample_genuine_ceo_english.wav",
        "description": "Authentic caller speech with natural prosody, vocal fold micro-jitter, and formants."
    })

    # Sample 2: AI Voice Clone Impersonating CEO (ElevenLabs Deep Clone)
    pcm2, float2 = generate_ai_voice_clone(duration_sec=3.5, base_f0=120.0, vocoder_type="ElevenLabs Voice Clone")
    path2 = os.path.join(SAMPLE_DIR, "sample_cloned_ceo_impersonation.wav")
    wav.write(path2, SAMPLE_RATE, pcm2)
    samples_metadata.append({
        "id": "sample_cloned_ceo_impersonation",
        "title": "AI Deep Clone - CEO Fund Transfer Attack",
        "language": "Indian English",
        "type": "SYNTHETIC_CLONE",
        "speaker": "Impersonating CEO (Rajesh Sharma)",
        "file_path": "data/sample_audio/sample_cloned_ceo_impersonation.wav",
        "description": "AI-cloned voice executing wire transfer attack. Exhibits high-frequency vocoder phase artifacts and pitch jump."
    })

    # Sample 3: Genuine CFO Speech (Hindi Dialect)
    pcm3, float3 = generate_natural_human_voice(duration_sec=4.0, base_f0=175.0, speaker_name="Ananya Sen (CFO)")
    path3 = os.path.join(SAMPLE_DIR, "sample_genuine_cfo_hindi.wav")
    wav.write(path3, SAMPLE_RATE, pcm3)
    samples_metadata.append({
        "id": "sample_genuine_cfo_hindi",
        "title": "Ananya Sen (CFO) - Genuine Hindi Call",
        "language": "Hindi / Hinglish",
        "type": "GENUINE",
        "speaker": "Ananya Sen (CFO)",
        "file_path": "data/sample_audio/sample_genuine_cfo_hindi.wav",
        "description": "Genuine female executive voice with natural pitch dynamics in Hindi context."
    })

    # Sample 4: RVC AI Voice Clone (Hindi OTP Fraud Attack)
    pcm4, float4 = generate_ai_voice_clone(duration_sec=3.8, base_f0=175.0, vocoder_type="RVC Hindi Voice Clone")
    path4 = os.path.join(SAMPLE_DIR, "sample_cloned_cfo_otp_attack.wav")
    wav.write(path4, SAMPLE_RATE, pcm4)
    samples_metadata.append({
        "id": "sample_cloned_cfo_otp_attack",
        "title": "AI Voice Clone - CFO Hindi OTP Social Engineering",
        "language": "Hindi",
        "type": "SYNTHETIC_CLONE",
        "speaker": "Impersonating CFO (Ananya Sen)",
        "file_path": "data/sample_audio/sample_cloned_cfo_otp_attack.wav",
        "description": "Manipulated synthetic voice attempting urgent OTP verification override."
    })

    # Write samples catalog index
    with open(os.path.join(SAMPLE_DIR, "samples_catalog.json"), "w") as f:
        json.dump(samples_metadata, f, indent=2)

    # Enroll Reference Voiceprints
    print("Enrolling reference CXO voiceprints into database...")
    from app.core.biometric_verifier import BiometricVerifier
    verifier = BiometricVerifier(enrolled_dir=VOICEPRINT_DIR)
    
    # Enroll Rajesh Sharma (CEO) using his genuine sample
    verifier.enroll_voiceprint(
        speaker_id="spk_ceo_rajesh",
        name="Rajesh Sharma",
        role="Chief Executive Officer (CEO)",
        audio_data=float1
    )

    # Enroll Ananya Sen (CFO) using her genuine sample
    verifier.enroll_voiceprint(
        speaker_id="spk_cfo_ananya",
        name="Ananya Sen",
        role="Chief Financial Officer (CFO)",
        audio_data=float3
    )

    print("Sample Dataset & Enrolled Voiceprints built successfully!")

if __name__ == "__main__":
    build_dataset()
