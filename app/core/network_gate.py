import numpy as np
import re
from typing import Dict, Any, Optional

class NetworkGate:
    """
    Layer 0: Telecom Network Signaling & Early-Media Interception Gate.
    Inspects SIP signaling, STIR/SHAKEN caller authentication, CLI spoofing, 
    VoIP proxy anomalies, and early-media (0.5s - 1.5s pre-answer snippet) 
    synthetic voice bursts before 2-way audio connection is established.
    """
    def __init__(self):
        # Known suspicious VoIP software signatures or high-risk SIP user agents
        self.suspicious_user_agents = [
            r'sip-tester', r'sipp', r'asterisk', r'freepbx', r'custom-sip-bot',
            r'python-requests', r'go-sip', r'voip-script', r'spoofed-agent'
        ]
        
    def analyze_sip_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Inspect SIP INVITE headers, STIR/SHAKEN tokens, and User-Agent headers.
        """
        headers = headers or {}
        user_agent = headers.get("User-Agent", "Mozilla/5.0 WebRTC/1.0")
        stir_shaken_passport = headers.get("Identity", "")
        cli_number = headers.get("From-CLI", "+1-800-555-0199")
        ip_origin = headers.get("X-Origin-IP", "192.168.1.100")
        is_voip_proxy = headers.get("X-Is-VoIP-Proxy", "false").lower() == "true"
        
        # 1. Check STIR/SHAKEN Attestation
        # 'A' = Full attestation (verified caller identity & right to use number)
        # 'B' = Partial attestation (verified caller, but unverified right to number)
        # 'C' = Gateway attestation (unverified caller & origin - HIGH RISK for spoofing)
        attestation = headers.get("STIR-SHAKEN-Attestation", "A")
        
        attestation_score = 0.0
        if attestation == "A":
            attestation_score = 0.05
        elif attestation == "B":
            attestation_score = 0.40
        elif attestation == "C" or attestation == "INVALID":
            attestation_score = 0.90
            
        # 2. Check User-Agent anomaly
        ua_risk = 0.0
        for pattern in self.suspicious_user_agents:
            if re.search(pattern, user_agent, re.IGNORECASE):
                ua_risk = 0.85
                break
                
        # 3. Check CLI Spoofing & VoIP Proxy
        proxy_risk = 0.85 if is_voip_proxy else 0.10
        
        # Combine Network Signaling Score
        signaling_score = max(attestation_score, ua_risk, proxy_risk * 0.7)
        
        return {
            "signaling_risk_score": float(np.round(signaling_score, 3)),
            "attestation": attestation,
            "attestation_score": attestation_score,
            "user_agent": user_agent,
            "user_agent_risk": ua_risk,
            "is_voip_proxy": is_voip_proxy,
            "proxy_risk": proxy_risk,
            "cli_number": cli_number,
            "ip_origin": ip_origin
        }

    def inspect_early_media(
        self, 
        audio_snippet: Optional[np.ndarray], 
        sample_rate: int = 16000,
        spectral_analyzer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Screen early-media audio (SIP 183 Session Progress pre-answer ring/greeting snippet)
        for neural vocoder signatures before opening 2-way audio communication.
        """
        if audio_snippet is None or len(audio_snippet) < sample_rate * 0.2:
            return {
                "early_media_analyzed": False,
                "early_media_risk": 0.0,
                "reason": "Insufficient early-media duration"
            }
            
        # Check energy level to ignore silence
        rms = np.sqrt(np.mean(audio_snippet ** 2))
        if rms < 0.008:
            return {
                "early_media_analyzed": True,
                "early_media_risk": 0.05,
                "reason": "Early-media background silence verified"
            }
            
        # Perform quick high-frequency phase anomaly check if spectral analyzer is provided
        early_risk = 0.05
        phase_anomaly = False
        if spectral_analyzer is not None:
            spec_res = spectral_analyzer.extract_spectral_features(audio_snippet)
            early_risk = float(spec_res.get("synthetic_spectral_score", 0.05))
            phase_anomaly = (spec_res.get("phase_jitter", 1.35) < 1.25) or (spec_res.get("high_freq_ratio", 0.0) > 0.055)
            
        return {
            "early_media_analyzed": True,
            "early_media_risk": float(np.round(early_risk, 3)),
            "phase_anomaly": phase_anomaly,
            "rms_energy": float(np.round(rms, 4))
        }

    def evaluate_pre_call_gate(
        self, 
        headers: Optional[Dict[str, str]] = None,
        audio_snippet: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        spectral_analyzer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate full Layer 0 Pre-Call Interception Gate.
        Determines whether to DROP/BLOCK call before 2-way communication is connected.
        """
        sip_eval = self.analyze_sip_headers(headers)
        early_media_eval = self.inspect_early_media(audio_snippet, sample_rate, spectral_analyzer)
        
        signaling_risk = sip_eval["signaling_risk_score"]
        early_media_risk = early_media_eval.get("early_media_risk", 0.0)
        
        # Combined Network & Early-Media Interception Risk (0.0 to 1.0)
        total_network_risk = max(signaling_risk, early_media_risk)
        
        # Pre-call drop threshold (e.g. >= 0.75 drops call prior to 2-way audio channel open)
        is_blocked = total_network_risk >= 0.75
        
        block_reasons = []
        if sip_eval["attestation"] in ["C", "INVALID"]:
            block_reasons.append(f"STIR/SHAKEN Attestation '{sip_eval['attestation']}' (CLI Spoofing Suspect)")
        if sip_eval["user_agent_risk"] > 0.5:
            block_reasons.append(f"Suspicious SIP User-Agent: {sip_eval['user_agent']}")
        if sip_eval["is_voip_proxy"]:
            block_reasons.append("High-risk anonymized VoIP proxy node detected")
        if early_media_eval.get("phase_anomaly", False) or early_media_risk > 0.7:
            block_reasons.append("Early-media vocoder phase anomaly detected in pre-answer snippet")
            
        reason_str = "; ".join(block_reasons) if block_reasons else "Clean telecommunication signaling & early-media"
        
        return {
            "blocked": is_blocked,
            "action": "DROP_BEFORE_2WAY_COMMUNICATION" if is_blocked else "ALLOW_2WAY_CONNECTION",
            "network_risk_score": float(np.round(total_network_risk, 3)),
            "reason": reason_str,
            "sip_details": sip_eval,
            "early_media_details": early_media_eval
        }
