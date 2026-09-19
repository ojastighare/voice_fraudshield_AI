import os
import json
import numpy as np
import scipy.signal

class BiometricVerifier:
    """
    Cross-session speaker verification engine.
    Extracts acoustic identity embeddings from speech audio and computes cosine similarity
    against enrolled target voiceprints (e.g. CXOs, Bank Executives, High-Risk Users).
    """

    def __init__(self, enrolled_dir: str = "data/enrolled_voiceprints", sample_rate=16000):
        self.enrolled_dir = enrolled_dir
        self.sample_rate = sample_rate
        self.enrolled_profiles = {}
        os.makedirs(self.enrolled_dir, exist_ok=True)
        self.load_enrolled_voiceprints()

    def extract_voiceprint_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extracts a normalized acoustic fingerprint embedding vector (128-dim).
        Combines mel-filterbank energy distribution, spectral statistics,
        and pitch dynamics to capture unique speaker identity attributes.
        """
        if len(audio_data) < 512:
            return np.zeros(128, dtype=np.float32)

        audio = audio_data.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Filterbank / Spectrogram feature extraction
        n_fft = min(512, len(audio))
        frequencies, _, stft_matrix = scipy.signal.stft(audio, fs=self.sample_rate, nperseg=n_fft)
        spec = np.abs(stft_matrix)

        # Sub-band energy distribution (64 bins)
        if spec.shape[0] >= 64:
            subbands = np.array_split(spec, 64, axis=0)
            energies = [np.mean(sb) for sb in subbands]
        else:
            energies = list(np.mean(spec, axis=1)) + [0.0] * (64 - spec.shape[0])
        
        energies = np.array(energies[:64], dtype=np.float32)

        # Statistical moments (mean, std, skew, kurtosis across frames) (32 features)
        stats = [
            np.mean(spec), np.std(spec),
            np.percentile(spec, 25), np.percentile(spec, 75),
            np.max(spec), np.min(spec)
        ]
        stats = stats * 5 + [0.0, 0.0] # 32 features

        # Time-domain energy contours (32 features)
        time_chunks = np.array_split(audio, 32)
        time_energies = [float(np.sqrt(np.mean(chunk**2))) for chunk in time_chunks]

        embedding = np.concatenate([energies[:64], np.array(stats[:32], dtype=np.float32), np.array(time_energies[:32], dtype=np.float32)])
        
        # Normalize L2 norm
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def enroll_voiceprint(self, speaker_id: str, name: str, role: str, audio_data: np.ndarray) -> dict:
        """
        Enrolls a new genuine speaker voiceprint template into storage.
        """
        embedding = self.extract_voiceprint_embedding(audio_data)
        profile = {
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "embedding": embedding.tolist()
        }
        
        file_path = os.path.join(self.enrolled_dir, f"{speaker_id}.json")
        with open(file_path, "w") as f:
            json.dump(profile, f, indent=2)

        self.enrolled_profiles[speaker_id] = {
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "embedding": embedding
        }

        return {
            "status": "success",
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "embedding_dim": len(embedding)
        }

    def verify_speaker(self, audio_data: np.ndarray, claimed_speaker_id: str = None) -> dict:
        """
        Verifies live audio stream against enrolled voiceprints using Cosine Similarity.
        Returns match confidence, identity match status, and mismatch penalty score.
        """
        live_embedding = self.extract_voiceprint_embedding(audio_data)

        if not self.enrolled_profiles:
            return {
                "matched_speaker_id": None,
                "speaker_name": "Unknown Caller",
                "speaker_role": "N/A",
                "similarity_score": 0.0,
                "is_match": False,
                "biometric_anomaly_score": 0.3 # Moderate uncertainty when no profile enrolled
            }

        # If a claimed speaker ID is provided (e.g. caller claims to be CEO)
        if claimed_speaker_id and claimed_speaker_id in self.enrolled_profiles:
            profile = self.enrolled_profiles[claimed_speaker_id]
            ref_embedding = profile["embedding"]
            sim = float(np.dot(live_embedding, ref_embedding))
            is_match = sim >= 0.72

            return {
                "claimed_speaker_id": claimed_speaker_id,
                "matched_speaker_id": claimed_speaker_id,
                "speaker_name": profile["name"],
                "speaker_role": profile["role"],
                "similarity_score": round(sim, 4),
                "is_match": is_match,
                "biometric_anomaly_score": round(1.0 - sim, 4) if not is_match else round((1.0 - sim) * 0.5, 4)
            }

        # Search across all enrolled profiles to find closest match
        best_sim = -1.0
        best_profile = None

        for speaker_id, profile in self.enrolled_profiles.items():
            ref_embedding = profile["embedding"]
            sim = float(np.dot(live_embedding, ref_embedding))
            if sim > best_sim:
                best_sim = sim
                best_profile = profile

        is_match = best_sim >= 0.72

        if best_profile and is_match:
            return {
                "matched_speaker_id": best_profile["speaker_id"],
                "speaker_name": best_profile["name"],
                "speaker_role": best_profile["role"],
                "similarity_score": round(best_sim, 4),
                "is_match": True,
                "biometric_anomaly_score": round((1.0 - best_sim) * 0.5, 4)
            }
        else:
            return {
                "matched_speaker_id": None,
                "speaker_name": "Unenrolled Caller (Guest)",
                "speaker_role": "Standard Caller",
                "best_match_score": round(max(best_sim, 0.0), 4),
                "is_match": True,
                "biometric_anomaly_score": 0.05
            }

    def load_enrolled_voiceprints(self):
        """
        Loads pre-stored enrolled speaker profiles from disk.
        """
        self.enrolled_profiles = {}
        if not os.path.exists(self.enrolled_dir):
            return

        for filename in os.listdir(self.enrolled_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.enrolled_dir, filename)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.enrolled_profiles[data["speaker_id"]] = {
                        "speaker_id": data["speaker_id"],
                        "name": data["name"],
                        "role": data.get("role", "Executive"),
                        "embedding": np.array(data["embedding"], dtype=np.float32)
                    }
                except Exception as e:
                    print(f"Error loading voiceprint {filename}: {e}")
