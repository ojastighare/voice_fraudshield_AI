import numpy as np
import scipy.signal
import scipy.stats

class SpectralAnalyzer:
    """
    Analyzes acoustic and spectral signatures of audio signals to detect
    deep learning synthesis artifacts, vocoder phase inconsistencies,
    and high-frequency spectral roll-off anomalies.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_spectral_features(self, audio_data: np.ndarray) -> dict:
        """
        Extracts spectral features and computes synthetic artifact likelihood.
        """
        if len(audio_data) == 0:
            return self._empty_result()

        # 1. Voice Activity / Signal Energy Check
        audio = audio_data.astype(np.float32)
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        
        # If true silence / zero audio input (< -50dB RMS), return safe 0 risk
        if rms_energy < 0.003:
            return self._empty_result()

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # 1. STFT Spectrogram Computation
        n_fft = min(1024, len(audio))
        hop_length = n_fft // 4
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        frequencies, times, stft_matrix = scipy.signal.stft(
            audio, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )
        magnitude = np.abs(stft_matrix) + 1e-10
        power_spec = magnitude ** 2

        # 2. Spectral Centroid
        freq_bins = frequencies.reshape(-1, 1)
        spectral_centroid = np.sum(freq_bins * magnitude, axis=0) / np.sum(magnitude, axis=0)
        mean_centroid = float(np.mean(spectral_centroid))

        # 3. Spectral Bandwidth
        centroid_diff = (freq_bins - spectral_centroid) ** 2
        spectral_bandwidth = np.sqrt(np.sum(centroid_diff * magnitude, axis=0) / np.sum(magnitude, axis=0))
        mean_bandwidth = float(np.mean(spectral_bandwidth))

        # 4. Spectral Roll-off (85% energy point)
        cum_power = np.cumsum(power_spec, axis=0)
        total_power = cum_power[-1, :]
        threshold_power = 0.85 * total_power
        rolloff_idx = np.argmax(cum_power >= threshold_power, axis=0)
        rolloff_freqs = frequencies[rolloff_idx]
        mean_rolloff = float(np.mean(rolloff_freqs))

        # 5. Spectral Flatness (Geometric Mean / Arithmetic Mean)
        geo_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
        arith_mean = np.mean(magnitude, axis=0)
        spectral_flatness = geo_mean / (arith_mean + 1e-10)
        mean_flatness = float(np.mean(spectral_flatness))

        # 6. High-Frequency Vocoder Power Leakage (> 5500 Hz)
        mid_mask = (frequencies >= 300) & (frequencies <= 3500)
        mid_energy = float(np.sum(power_spec[mid_mask, :]))

        hf_mask = frequencies > 5500
        hf_energy = float(np.sum(power_spec[hf_mask, :]))
        hf_power_ratio = float(hf_energy / (mid_energy + 1e-10))

        # 7. Phase Determinism (Unwrapped Phase 2nd Derivative)
        angle = np.angle(stft_matrix)
        unwrapped = np.unwrap(angle, axis=1)
        phase_jitter = float(np.mean(np.abs(np.diff(unwrapped, n=2, axis=1)))) if unwrapped.shape[1] > 2 else 1.35

        # 8. Frame-to-frame Spectral Flux
        spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)) if magnitude.shape[1] > 1 else np.array([0.0])
        mean_flux = float(np.mean(spectral_flux))

        # 9. Compute Synthetic Artifact Indicators
        # - Phase determinism: neural vocoder reconstruction creates deterministic phase paths (< 1.30)
        if phase_jitter < 1.30:
            phase_anomaly = float(np.clip((1.30 - phase_jitter) / 0.25, 0.0, 1.0))
        else:
            phase_anomaly = 0.0

        # - High-frequency vocoder power leakage (> 5500 Hz):
        if hf_power_ratio > 0.020:
            hf_anomaly = float(np.clip((hf_power_ratio - 0.020) / 0.035, 0.0, 1.0))
        else:
            hf_anomaly = 0.0

        # - Spectral Flatness Anomaly (Neural TTS exhibits unnaturally low flatness < 0.035 in speech)
        if mean_flatness < 0.035:
            flatness_anomaly = float(np.clip((0.035 - mean_flatness) / 0.025, 0.0, 1.0))
        else:
            flatness_anomaly = 0.0

        # - Spectral Bandwidth Discontinuity (Neural vocoder high-band dispersion > 1300 Hz)
        if mean_bandwidth > 1300.0 and (hf_anomaly > 0.10 or phase_anomaly > 0.10):
            bandwidth_anomaly = float(np.clip((mean_bandwidth - 1300.0) / 600.0, 0.0, 1.0))
        else:
            bandwidth_anomaly = 0.0

        synthetic_spectral_score = round(
            0.40 * phase_anomaly + 
            0.35 * hf_anomaly + 
            0.15 * flatness_anomaly + 
            0.10 * bandwidth_anomaly, 
            4
        )

        return {
            "spectral_centroid": round(mean_centroid, 2),
            "spectral_bandwidth": round(mean_bandwidth, 2),
            "spectral_rolloff": round(mean_rolloff, 2),
            "spectral_flatness": round(mean_flatness, 4),
            "high_freq_ratio": round(hf_power_ratio, 4),
            "phase_jitter": round(phase_jitter, 3),
            "spectral_flux": round(mean_flux, 4),
            "synthetic_spectral_score": round(synthetic_spectral_score, 4)
        }

    def _empty_result(self):
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "high_freq_ratio": 0.0,
            "spectral_flux": 0.0,
            "synthetic_spectral_score": 0.0
        }

def clip_score(val, min_val=0.0, max_val=1.0):
    return float(np.clip(val, min_val, max_val))
