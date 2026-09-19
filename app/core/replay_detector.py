import numpy as np
import scipy.signal

class ReplayDetector:
    """
    Replay Attack & Acoustic Distortion Detector.
    Identifies acoustic signatures of audio playback (e.g. playing a pre-recorded
    voice sample through a smartphone/laptop loudspeaker into another microphone).
    Indicators: Loudspeaker bandpass cutoff, secondary room impulse response,
    channel noise floor, and spectral crest factor anomalies.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def detect_replay_attack(self, audio_data: np.ndarray) -> dict:
        """
        Analyzes audio for replay attack indicators.
        Returns replay probability score (0-100), suspicion flag, and detailed indicators.
        """
        if len(audio_data) < 512:
            return {
                "replay_score": 0.0,
                "is_replay_suspected": False,
                "indicators": []
            }

        audio = audio_data.astype(np.float32)
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        
        # If true silence / negligible signal, return safe baseline
        if rms_energy < 0.003:
            return {
                "replay_score": 0.0,
                "is_replay_suspected": False,
                "indicators": []
            }

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        indicators = []
        # Clean authentic live speech has a healthy baseline presence of 4.0 - 8.0%
        score_accum = 4.0

        # 1. Loudspeaker Channel Frequency Distortion
        n_fft = min(1024, len(audio))
        hop_length = n_fft // 4
        freqs, _, stft_matrix = scipy.signal.stft(
            audio, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
        )
        mag = np.abs(stft_matrix) + 1e-10

        mid_mask = (freqs >= 600) & (freqs <= 3500)
        mid_energy = float(np.mean(mag[mid_mask, :])) + 1e-10

        hf_mask = freqs > 7200
        hf_energy = float(np.mean(mag[hf_mask, :])) if np.any(hf_mask) else 0.0
        hf_ratio = hf_energy / mid_energy

        # Loudspeakers severely roll off > 7.2kHz combined with high mid-range cavity distortion
        if hf_ratio < 0.003 and np.mean(mag[freqs < 400, :]) / mid_energy < 0.04:
            score_accum += 25.0
            indicators.append("Double Channel Filtering / Micro-Speaker Roll-Off")

        # 2. Secondary Room Impulse Response (Decoupled from Voice Pitch Periodicity)
        # High-pitch voices have natural harmonic autocorrelation peaks at multiples of pitch period T0.
        # We find non-harmonic secondary reflection peaks in the 15ms - 50ms window.
        corr = scipy.signal.correlate(audio, audio, mode='full')
        corr = corr[len(audio) - 1:]
        corr = corr / (corr[0] + 1e-10)

        # Estimate primary pitch lag T0 in the 1.6ms (600Hz) to 20ms (50Hz) search window
        search_start = int(0.0016 * self.sample_rate)
        search_end = int(0.020 * self.sample_rate)
        pitch_lag = int(np.argmax(corr[search_start:search_end]) + search_start) if len(corr) > search_end else 0

        lag_15ms = int(0.015 * self.sample_rate)
        lag_50ms = int(0.050 * self.sample_rate)
        if len(corr) > lag_50ms:
            sub_corr = np.copy(corr[lag_15ms:lag_50ms])
            # Mask out harmonic integer multiples of the pitch lag (within +/- 5 samples)
            if pitch_lag > 0:
                for k in range(1, 15):
                    h_idx = k * pitch_lag - lag_15ms
                    if 0 <= h_idx < len(sub_corr):
                        sub_corr[max(0, h_idx - 5):min(len(sub_corr), h_idx + 6)] = 0.0

            non_pitch_peak = float(np.max(np.abs(sub_corr))) if len(sub_corr) > 0 else 0.0
            if non_pitch_peak > 0.45:
                score_accum += 30.0
                indicators.append("Non-Harmonic Multipath Room Reflection Signature")
            elif non_pitch_peak > 0.32:
                score_accum += 15.0

        # 3. Channel Background Noise Floor (Evaluated strictly during speech pauses)
        energy = audio ** 2
        silent_frames = energy[energy < 0.01]
        if len(silent_frames) > 80:
            pause_noise = float(np.percentile(silent_frames, 50))
            if pause_noise > 0.004:
                score_accum += 20.0
                indicators.append("Elevated Loudspeaker DAC Noise Floor / Ambient Loop")

        replay_score = float(min(round(score_accum, 1), 100.0))
        is_replay_suspected = replay_score >= 50.0

        return {
            "replay_score": replay_score,
            "is_replay_suspected": is_replay_suspected,
            "indicators": indicators
        }
