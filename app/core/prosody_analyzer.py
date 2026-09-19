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
        f0_contour = self._extract_f0_autocorr(audio)
        voiced_f0 = f0_contour[f0_contour > 0]

        if len(voiced_f0) > 5:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            min_f0 = float(np.min(voiced_f0))
            max_f0 = float(np.max(voiced_f0))
            pitch_range = max_f0 - min_f0

            # 2. Vocal Jitter (Cycle-to-cycle frequency variation)
            f0_diffs = np.abs(np.diff(voiced_f0))
            jitter = float(np.mean(f0_diffs) / (mean_f0 + 1e-5))

            # 3. Vocal Shimmer (Cycle-to-cycle amplitude variation)
            frame_len = int(0.025 * self.sample_rate)
            hop_len = int(0.010 * self.sample_rate)
            amplitudes = []
            for i in range(0, len(audio) - frame_len, hop_len):
                frame = audio[i:i + frame_len]
                amplitudes.append(np.max(np.abs(frame)))
            amplitudes = np.array(amplitudes)
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer = float(np.mean(amp_diffs) / (np.mean(amplitudes) + 1e-5))

            # 4. Unnatural Pitch Jump Detection
            # Neural vocoder synthesis glitches create abrupt relative octave steps (>65% jump)
            rel_diffs = f0_diffs / (voiced_f0[:-1] + 1e-5)
            pitch_jumps = float(np.sum(rel_diffs > 0.65) / len(voiced_f0))
        else:
            mean_f0, std_f0, jitter, shimmer, pitch_range, pitch_jumps = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 5. Silence & Speech Rhythm Micro-Pause Analysis
        energy = audio ** 2
        silence_threshold = 0.01
        is_silent = energy < silence_threshold
        silence_ratio = float(np.sum(is_silent) / len(audio))

        # 6. Compute Synthetic Prosody Score (0.0 to 1.0)
        # Neural TTS anomalies:
        # - Sudden discrete pitch spikes / glitches (relative pitch_jumps > 0.20)
        # - Extreme unnatural flutter / jitter (> 0.18)
        # - Rigid flat robotic pitch line (voiced_f0 > 40 with std_f0 < 2.0)
        prosodic_anomalies = []

        if len(voiced_f0) > 10:
            # Unnatural discrete pitch jumps
            prosodic_anomalies.append(float(np.clip(pitch_jumps * 8.0, 0.0, 1.0)))
            # Extreme flutter / jitter
            prosodic_anomalies.append(1.0 if jitter > 0.18 else 0.0)
            # Long-duration flat robotic line
            if len(voiced_f0) > 60 and std_f0 < 2.0:
                prosodic_anomalies.append(0.8)
            else:
                prosodic_anomalies.append(0.0)
        else:
            prosodic_anomalies.append(0.0)

        synthetic_prosody_score = float(np.mean(prosodic_anomalies))

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

    def _extract_f0_autocorr(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract F0 pitch contour using frame-wise autocorrelation supporting 50Hz - 600Hz.
        """
        frame_len = int(0.03 * self.sample_rate) # 30ms
        hop_len = int(0.015 * self.sample_rate)  # 15ms
        min_lag = int(self.sample_rate / 600)    # max F0 600 Hz (supports high-pitch voices)
        max_lag = int(self.sample_rate / 50)     # min F0 50 Hz

        f0_list = []

        for i in range(0, len(audio) - frame_len, hop_len):
            frame = audio[i:i + frame_len]
            if np.max(np.abs(frame)) < 0.02:
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

        return np.array(f0_list)

    def _empty_result(self):
        return {
            "mean_f0": 0.0,
            "f0_std": 0.0,
            "pitch_range": 0.0,
            "jitter": 0.0,
            "shimmer": 0.0,
            "unnatural_pitch_jumps": 0.0,
            "silence_ratio": 0.0,
            "synthetic_prosody_score": 0.0
        }
