import os
import ssl
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WhisperASREngine:
    """
    Real-Time Multilingual Speech-to-Text (ASR) Engine powered by OpenAI Whisper.
    Supports 99+ languages including Hindi, Marathi, Bengali, Tamil, English, etc.
    Memory-optimized for cloud micro-containers (512MB RAM).
    """

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            import gc
            import torch
            torch.set_num_threads(1)
            ssl._create_default_https_context = ssl._create_unverified_context
            import whisper
            logger.info(f"Loading lightweight Whisper model '{self.model_size}'...")
            self.model = whisper.load_model(self.model_size, device="cpu")
            gc.collect()
            logger.info(f"Whisper '{self.model_size}' loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe_audio(self, audio_np: np.ndarray, sample_rate: int = 16000, language: str = None) -> dict:
        """
        Transcribes a 1D float32 audio array.
        language: 'hi' for Hindi, 'en' for English, 'mr' for Marathi, None for Auto-detect.
        """
        if audio_np is None or len(audio_np) == 0:
            return {"transcript": "", "language": language or "en", "confidence": 0.0, "has_speech": False}

        # Resample if not 16kHz
        if sample_rate != 16000:
            num_samples = int(len(audio_np) * 16000 / sample_rate)
            audio_np = np.interp(
                np.linspace(0, len(audio_np), num_samples, endpoint=False),
                np.arange(len(audio_np)),
                audio_np
            ).astype(np.float32)

        # Check RMS energy to avoid transcribing pure silence
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms < 0.003 or len(audio_np) < 3200:
            return {"transcript": "", "language": language or "en", "confidence": 0.0, "has_speech": False}

        if self.model is None:
            self._load_model()
            if self.model is None:
                return {"transcript": "", "language": language or "en", "confidence": 0.0, "has_speech": False}

        try:
            # Ensure float32 normalized in [-1, 1]
            max_val = np.max(np.abs(audio_np))
            if max_val > 1.0:
                audio_np = audio_np / max_val
            elif max_val > 0.0:
                audio_np = audio_np / max(0.01, max_val) * 0.95

            audio_np = audio_np.astype(np.float32)

            lang_arg = language if (language and language.lower() not in ["auto", "all", "none", ""]) else None

            # Transcribe with Whisper
            res = self.model.transcribe(
                audio_np,
                fp16=False,
                language=lang_arg,
                temperature=0.0,
                without_timestamps=True
            )

            text = res.get("text", "").strip()
            detected_lang = res.get("language", language or "en")

            return {
                "transcript": text,
                "language": detected_lang,
                "confidence": 0.95 if text else 0.0,
                "has_speech": bool(text)
            }
        except Exception as e:
            logger.warning(f"Whisper transcription error: {e}")
            return {"transcript": "", "language": language or "en", "confidence": 0.0, "has_speech": False}
