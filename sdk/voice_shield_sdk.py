import requests
import json
import base64
from typing import Dict, Any, Optional

class VoiceShieldSDK:
    """
    VoiceShield AI Enterprise Integration SDK.
    Enables banking applications, telephony servers, and enterprise communication systems
    to integrate real-time voice clone detection in 3 lines of code.
    """

    def __init__(self, endpoint_url: str = "http://localhost:8000"):
        self.endpoint_url = endpoint_url.rstrip("/")

    def verify_audio_file(
        self,
        file_path: str,
        claimed_speaker_id: Optional[str] = None,
        transaction_amount: float = 0.0,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Verifies audio file for synthetic voice cloning and impersonation threats.
        """
        url = f"{self.endpoint_url}/api/v1/analyze"
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "claimed_speaker_id": claimed_speaker_id or "",
                "transaction_amount": str(transaction_amount),
                "transcript": transcript
            }
            res = requests.post(url, files=files, data=data)
            res.raise_for_status()
            return res.json()

    def verify_raw_audio_bytes(
        self,
        audio_bytes: bytes,
        claimed_speaker_id: Optional[str] = None,
        transaction_amount: float = 0.0,
        transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Verifies raw PCM or WAV audio bytes.
        """
        url = f"{self.endpoint_url}/api/v1/analyze-base64"
        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        payload = {
            "audio_base64": b64_audio,
            "claimed_speaker_id": claimed_speaker_id,
            "transaction_amount": transaction_amount,
            "transcript": transcript
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json()

    def enroll_voiceprint(self, speaker_id: str, name: str, role: str, audio_file_path: str) -> Dict[str, Any]:
        """
        Enrolls a new genuine speaker voiceprint baseline.
        """
        url = f"{self.endpoint_url}/api/v1/enroll"
        with open(audio_file_path, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "speaker_id": speaker_id,
            "name": name,
            "role": role,
            "audio_base64": b64_audio
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json()
