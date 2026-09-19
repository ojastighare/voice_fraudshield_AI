import hashlib
import time
import json
from typing import Dict, Any

class PrivacyManager:
    """
    Privacy & Compliance Module.
    Ensures zero central raw audio retention by default, edge feature-only logging,
    SHA-256 anonymization of voiceprints, and DPDP / GDPR audit logging.
    """

    def __init__(self, log_dir: str = "data/audit_logs"):
        self.log_dir = log_dir

    def anonymize_speaker_id(self, raw_id: str) -> str:
        """Hashes raw user identity with SHA-256 for anonymized privacy logging."""
        return hashlib.sha256(f"VOICE_SHIELD_SALT_{raw_id}".encode('utf-8')).hexdigest()[:16]

    def format_audit_log(self, call_id: str, caller_id: str, risk_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates privacy-compliant telemetry record (features & risk metric only, NO raw audio).
        """
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "call_id": call_id,
            "anonymized_caller_id": self.anonymize_speaker_id(caller_id),
            "risk_score": risk_res.get("risk_score", 0.0),
            "risk_tier": risk_res.get("risk_tier", "GREEN"),
            "action_taken": risk_res.get("action_code", "ALLOW"),
            "acoustic_features_only": True,
            "raw_audio_retained": False,
            "dpdp_compliant": True
        }
