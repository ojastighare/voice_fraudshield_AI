import re
from typing import Dict, Any, List

class RiskScoringEngine:
    """
    Real-Time Dynamic Impersonation Risk Scoring Engine.
    Aggregates multi-layer acoustic, prosodic, biometric, and contextual inputs
    to compute a real-time Risk Index (0 - 100) and actionable security recommendations.
    """

    def __init__(self):
        # Default policy thresholds
        self.amber_threshold = 40.0
        self.red_threshold = 70.0

        # High-risk financial and social engineering trigger phrases
        self.sensitive_keywords = [
            r"\bwire transfer\b", r"\bfund transfer\b", r"\botp\b", r"\bpin\b",
            r"\bpassword\b", r"\burgent\b", r"\bemergency\b", r"\bbank account\b",
            r"\brouting number\b", r"\bswift\b", r"\bauthorize\b", r"\boverride\b",
            r"\bcrore\b", r"\blakh\b", r"\btransfer rupees\b"
        ]

    def set_policy_thresholds(self, amber: float, red: float):
        self.amber_threshold = float(amber)
        self.red_threshold = float(red)

    def calculate_risk(
        self,
        spectral_res: Dict[str, Any],
        prosody_res: Dict[str, Any],
        biometric_res: Dict[str, Any],
        deep_ml_res: Dict[str, Any],
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Computes real-time dynamic risk score (0-100), risk level, and mitigation policy triggers.
        """
        context_data = context_data or {}

        # 1. Base Layer Scores (0.0 to 1.0)
        s_spec = spectral_res.get("synthetic_spectral_score", 0.0)
        s_pros = prosody_res.get("synthetic_prosody_score", 0.0)
        s_ml = deep_ml_res.get("synthetic_probability", 0.0)
        s_bio = biometric_res.get("biometric_anomaly_score", 0.3)

        # 2. Acoustic Multi-layer Score (0.0 to 1.0)
        acoustic_synthetic_score = (
            0.35 * s_spec +
            0.25 * s_pros +
            0.40 * s_ml
        )

        # 3. Biometric Identity Penalty
        # If caller claims identity but fails biometric similarity check
        is_biometric_mismatch = (
            biometric_res.get("claimed_speaker_id") is not None and
            not biometric_res.get("is_match", True)
        )
        bio_penalty = 0.35 if is_biometric_mismatch else 0.0

        # 4. Contextual Risk Factor Enrichment
        context_risk_boost = 0.0
        context_flags = []

        # Check transaction amount
        tx_amount = context_data.get("transaction_amount", 0.0)
        if tx_amount > 100000: # > 1 Lakh INR
            context_risk_boost += 12.0
            context_flags.append(f"High-Value Transaction Context (₹{tx_amount:,.2f})")
        elif tx_amount > 500000:
            context_risk_boost += 20.0
            context_flags.append(f"Critical Transaction Context (₹{tx_amount:,.2f})")

        # Check conversation text transcript (if speech-to-text provided)
        transcript = context_data.get("transcript", "").lower()
        matched_phrases = []
        if transcript:
            for kw_pattern in self.sensitive_keywords:
                if re.search(kw_pattern, transcript):
                    matched_phrases.append(kw_pattern.replace(r"\b", ""))
            
            if matched_phrases:
                context_risk_boost += min(len(matched_phrases) * 8.0, 25.0)
                context_flags.append(f"High-Risk Phrasal Triggers Detected: {', '.join(matched_phrases)}")

        # Telephony origin anomaly
        if context_data.get("voip_proxy_detected", False) or context_data.get("spoofed_cli", False):
            context_risk_boost += 15.0
            context_flags.append("VoIP / Call CLI Spoofing Anomaly Detected")

        # 5. Calculate Final Risk Score (0 to 100)
        raw_score = (acoustic_synthetic_score * 70.0) + (bio_penalty * 100.0) + context_risk_boost
        risk_score = round(float(min(max(raw_score, 0.0), 100.0)), 1)

        # 6. Risk Level Categorization
        if risk_score >= self.red_threshold:
            risk_tier = "RED"
            risk_label = "CRITICAL RISK - AI Voice Clone Impersonation Detected"
            action_code = "BLOCK_AND_ALERT"
            recommended_actions = [
                "IMMEDIATE ACTION: Halt fund transfer / sensitive data disclosure.",
                "Trigger mandatory Step-Up Multi-Factor Authentication (SMS OTP + Push App Approve).",
                "Initiate mandatory manual call-back on registered phone number.",
                "Escalate incident alert to Enterprise Fraud Operations Team."
            ]
        elif risk_score >= self.amber_threshold:
            risk_tier = "AMBER"
            risk_label = "SUSPICIOUS - Potential Voice Distortion / Unverified Identity"
            action_code = "VERIFY_BEFORE_PROCEED"
            recommended_actions = [
                "WARNING: Prompt frontline operator to verify caller identity.",
                "Request caller to state secondary security question / passphrase.",
                "Enable dual-operator authorization for pending requests."
            ]
        else:
            risk_tier = "GREEN"
            risk_label = "SAFE - Genuine Human Voice Verified"
            action_code = "ALLOW"
            recommended_actions = [
                "Normal conversation flow verified.",
                "Continuous background integrity monitoring active."
            ]

        return {
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "risk_label": risk_label,
            "action_code": action_code,
            "acoustic_synthetic_score": round(acoustic_synthetic_score, 4),
            "biometric_match": biometric_res.get("is_match", True),
            "speaker_name": biometric_res.get("speaker_name", "Unknown Caller"),
            "speaker_role": biometric_res.get("speaker_role", "N/A"),
            "biometric_similarity": biometric_res.get("similarity_score", 0.0),
            "context_flags": context_flags,
            "recommended_actions": recommended_actions,
            "breakdown": {
                "spectral_artifact_score": s_spec,
                "prosody_artifact_score": s_pros,
                "deep_ml_synthetic_prob": s_ml,
                "biometric_anomaly_score": s_bio
            }
        }
