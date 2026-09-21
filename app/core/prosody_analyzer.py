import numpy as np
import scipy.signal

class ProsodyAnalyzer:
    """
    Analyzes speech rhythm, pitch contours (F0), pitch instability (jitter),
    amplitude instability (shimmer), and pause micro-variations.
    Neural TTS systems often struggle with natural micro-tremors and natural human prosody.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def analyze_prosody(self, audio_data: np.ndarray) -> dict:
        """
        Calculates prosodic features and computes synthetic prosody score.
        """
        if len(audio_data) < 512:
            return self._empty_result()

        audio = audio_data.astype(np.float32)
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        
        # If true silence / zero audio input (< -50dB RMS), return safe 0 risk
        if rms_energy < 0.003:
            return self._empty_result()

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 1. Pitch Tracking via Autocorrelation (F0 extraction supporting 50Hz to 600Hz)
        frame_len = int(0.03 * self.sample_rate)
        hop_len = int(0.015 * self.sample_rate)
        f0_contour, amplitudes = self._extract_f0_and_amplitudes(audio, frame_len, hop_len)
        
        voiced_indices = np.where(f0_contour > 0)[0]
        voiced_f0 = f0_contour[voiced_indices]
        voiced_amps = amplitudes[voiced_indices] if len(amplitudes) == len(f0_contour) else amplitudes

        if len(voiced_f0) > 5:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            min_f0 = float(np.min(voiced_f0))
            max_f0 = float(np.max(voiced_f0))
            pitch_range = max_f0 - min_f0

            # 2. Vocal Jitter (Cycle-to-cycle frequency variation on voiced segments)
            f0_diffs = np.abs(np.diff(voiced_f0))
            jitter = float(np.mean(f0_diffs) / (mean_f0 + 1e-5))

            # 3. Vocal Shimmer (Cycle-to-cycle amplitude variation on voiced segments)
            if len(voiced_amps) > 2:
                amp_diffs = np.abs(np.diff(voiced_amps))
                shimmer = float(np.mean(amp_diffs) / (np.mean(voiced_amps) + 1e-5))
            else:
                shimmer = 0.05

            # 4. Unnatural Pitch Jump Detection (> 50% relative pitch step)
            rel_diffs = f0_diffs / (voiced_f0[:-1] + 1e-5)
            pitch_jumps = float(np.sum(rel_diffs > 0.50) / len(voiced_f0))
        else:
            mean_f0, std_f0, jitter, shimmer, pitch_range, pitch_jumps = 0.0, 0.0, 0.02, 0.05, 0.0, 0.0

        # 5. Silence & Speech Rhythm Micro-Pause Analysis
        energy = audio ** 2
        silence_threshold = 0.01
        is_silent = energy < silence_threshold
        silence_ratio = float(np.sum(is_silent) / len(audio))

        # 6. Compute Synthetic Prosody Score (0.0 to 1.0)
        # Neural TTS anomalies:
        # - Robotic hyper-smoothness (unnatural absence of vocal micro-tremors: shimmer < 0.025, jitter < 0.006)
        # - Step-wise pitch glitches (unnatural pitch jumps)
        # - Rigid flat robotic pitch contour (f0_std < 5.0)
        shimmer_anomaly = float(np.clip((0.025 - shimmer) / 0.020, 0.0, 1.0)) if (shimmer < 0.025 and len(voiced_f0) > 5) else 0.0
        jitter_anomaly = float(np.clip((0.006 - jitter) / 0.005, 0.0, 1.0)) if (jitter < 0.006 and len(voiced_f0) > 5) else 0.0
        pitch_jump_anomaly = float(np.clip(pitch_jumps * 6.0, 0.0, 1.0))
        flat_pitch_anomaly = 0.8 if (len(voiced_f0) > 20 and std_f0 < 4.0) else 0.0

        synthetic_prosody_score = round(
            0.45 * shimmer_anomaly + 
            0.30 * jitter_anomaly + 
            0.15 * pitch_jump_anomaly + 
            0.10 * flat_pitch_anomaly, 
            4
        )

        return {
            "mean_f0": round(mean_f0, 2),
            "f0_std": round(std_f0, 2),
            "pitch_range": round(pitch_range, 2),
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "unnatural_pitch_jumps": round(pitch_jumps, 4),
            "silence_ratio": round(silence_ratio, 4),
            "synthetic_prosody_score": round(synthetic_prosody_score, 4)
        }

    def _extract_f0_and_amplitudes(self, audio: np.ndarray, frame_len: int, hop_len: int) -> tuple:
        """
        Extract F0 pitch contour and frame amplitude envelopes supporting 50Hz - 600Hz.
        """
        min_lag = int(self.sample_rate / 600)    # max F0 600 Hz
        max_lag = int(self.sample_rate / 50)     # min F0 50 Hz

        f0_list = []
        amp_list = []

        for i in range(0, len(audio) - frame_len, hop_len):
            frame = audio[i:i + frame_len]
            amp = float(np.max(np.abs(frame)))
            amp_list.append(amp)
            
            if amp < 0.02:
                f0_list.append(0.0)
                continue

            # Autocorrelation
            corr = scipy.signal.correlate(frame, frame, mode='full')
            corr = corr[len(frame) - 1:]
            
            if len(corr) <= max_lag:
                f0_list.append(0.0)
                continue

            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                f0_list.append(0.0)
                continue

            peak_idx = np.argmax(search_region) + min_lag
            if corr[peak_idx] > 0.3 * corr[0]:
                f0 = self.sample_rate / peak_idx
                f0_list.append(f0)
            else:
                f0_list.append(0.0)

        return np.array(f0_list), np.array(amp_list)

    def _empty_result(self):
        return {
            "mean_f0": 0.0,
            "f0_std": 0.0,
            "pitch_range": 0.0,
            "jitter": 0.02,
            "shimmer": 0.05,
            "unnatural_pitch_jumps": 0.0,
            "silence_ratio": 0.0,
            "synthetic_prosody_score": 0.0
        }

