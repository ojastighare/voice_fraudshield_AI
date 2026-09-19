from typing import Dict, Any, List

class RiskFusionEngine:
    """
    Contextual Risk Fusion Engine.
    Combines 5 independent intelligence signals into a non-linear Overall Impersonation Risk Score:
    R = f(VoiceAuthenticity, SpeakerMismatch, SocialEngineering, TransactionRisk, ReplayRisk)
    Generates natural language Explainable AI summaries detailing exact reasons for warning flags.
    """

    def __init__(self):
        pass

    def fuse_risks(
        self,
        spectral_res: Dict[str, Any],
        prosody_res: Dict[str, Any],
        biometric_res: Dict[str, Any],
        deep_ml_res: Dict[str, Any],
        social_eng_res: Dict[str, Any],
        replay_res: Dict[str, Any],
        network_res: Dict[str, Any] = None,
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Fuses multi-layer intelligence metrics (Layer 0 Network Interception -> Layer 1 Acoustics -> Layer 2 Biometrics -> Layer 3 NLP) 
        into calibrated overall risk metric and explainability report.
        """
        network_res = network_res or {}
        context_data = context_data or {}

        # 0. Check Layer 0 Pre-Call Network Interception Gate
        network_blocked = network_res.get("blocked", False)
        network_risk_score = round(float(network_res.get("network_risk_score", 0.0) * 100.0), 1)

        # 1. Individual 5-Vector Metric Calculation (0 - 100%)
        # Metric 1: Voice Authenticity (Synthetic Score %)
        p_synthetic = deep_ml_res.get("synthetic_probability", 0.0)
        synth_spectral = spectral_res.get("synthetic_spectral_score", 0.0)
        synth_prosody = prosody_res.get("synthetic_prosody_score", 0.0)
        voice_authenticity_score = round(float(min(max((0.5 * p_synthetic + 0.3 * synth_spectral + 0.2 * synth_prosody) * 100.0, 0.0), 100.0)), 1)

        # Metric 2: Speaker Verification (Identity Match %)
        sim_score = biometric_res.get("similarity_score", 0.0)
        is_claimed = biometric_res.get("claimed_speaker_id") is not None
        is_match = biometric_res.get("is_match", True)

        if is_claimed:
            speaker_match_score = round(float(sim_score * 100.0), 1)
            speaker_mismatch_penalty = max((0.75 - sim_score) * 120.0, 30.0) if not is_match else 0.0
        else:
            # When caller identity is un-enrolled or no claimed ID passed
            speaker_match_score = 100.0 if is_match else round(float(max(sim_score, 0.45) * 100.0), 1)
            speaker_mismatch_penalty = 25.0 if (p_synthetic > 0.50 and not is_match) else 0.0

        # Metric 3: Social Engineering / Conversation Risk %
        se_score = social_eng_res.get("social_engineering_score", 0.0)
        se_tags = social_eng_res.get("tactic_tags", [])

        # Metric 4: Transaction Context Risk %
        tx_amount = float(context_data.get("transaction_amount", 0.0))
        tx_context = context_data.get("tx_context", {})
        if tx_amount <= 0 and tx_context.get("amount"):
            tx_amount = float(tx_context["amount"])

        if tx_amount >= 2000000: # >= 20 Lakhs
            transaction_context_risk = 95.0
        elif tx_amount >= 500000: # >= 5 Lakhs
            transaction_context_risk = 75.0
        elif tx_amount >= 100000: # >= 1 Lakh
            transaction_context_risk = 50.0
        elif tx_amount >= 25000: # >= 25k
            transaction_context_risk = 30.0
        elif tx_amount > 0 or tx_context.get("has_financial_intent"):
            transaction_context_risk = 20.0
        else:
            transaction_context_risk = 5.0 # Low active baseline

        # Metric 5: Replay Risk %
        replay_score = replay_res.get("replay_score", 4.0)

        # 2. Non-Linear Risk Fusion Function
        # Base weighted fusion
        base_fused = (
            0.35 * voice_authenticity_score +
            0.20 * se_score +
            0.15 * transaction_context_risk +
            0.20 * speaker_mismatch_penalty +
            0.10 * replay_score
        )

        # High-risk contextual multipliers & Primary AI Voice Clone Threat Amplifier
        multipliers = 0.0

        # Dedicated AI Voice Clone Detection Amplifier
        # High threat boost requires concurrence of acoustic and neural classifier
        if voice_authenticity_score >= 50.0 or p_synthetic >= 0.50:
            # Direct synthetic voice attack: Boost into RED Critical Risk
            synth_boost = max((voice_authenticity_score - 25.0) * 1.3, 45.0)
            multipliers += synth_boost
        elif voice_authenticity_score >= 35.0 and p_synthetic >= 0.35:
            multipliers += 15.0

        if network_blocked or network_risk_score >= 75.0:
            multipliers += 40.0 # Early network drop / CLI spoofing penalty
        elif network_risk_score >= 40.0:
            multipliers += 10.0

        if voice_authenticity_score > 50 and se_score > 35:
            multipliers += 15.0 # Combined synthetic audio + social engineering
        if is_claimed and not is_match and se_score > 30:
            multipliers += 20.0 # Speaker identity mismatch + urgent money request
        if "⚠ Secrecy / No-Callback" in se_tags or "⚠ Urgency & Time Pressure" in se_tags:
            multipliers += 10.0

        overall_risk_score = round(float(min(max(base_fused + multipliers, 0.0), 100.0)), 1)
        
        # Guardrails: Synthetic voice or pre-call drop must trigger high/critical risk
        if network_blocked:
            overall_risk_score = max(overall_risk_score, 94.0)
        elif (voice_authenticity_score >= 50.0 or p_synthetic >= 0.50):
            overall_risk_score = max(overall_risk_score, 88.0)
        elif voice_authenticity_score <= 35.0 and p_synthetic <= 0.30 and network_risk_score < 30.0 and se_score <= 30.0:
            # Genuine caller: Keep strictly in GREEN safe tier (4% - 10%)
            overall_risk_score = min(overall_risk_score, 9.5)

        # 3. Risk Tier & Adaptive Action Assignment
        if network_blocked or overall_risk_score >= 75.0:
            risk_tier = "RED"
            risk_label = "CRITICAL RISK - Pre-Call Drop / AI Voice Clone Impersonation" if network_blocked else "CRITICAL RISK - AI Voice Clone Impersonation Attack"
            action_code = "DROP_BEFORE_2WAY_COMMUNICATION" if network_blocked else "EMERGENCY_FREEZE_PAYMENT"
            adaptive_threshold_msg = "Layer 0 Pre-Call Interception Gate: Call blocked before 2-way audio channel established!" if network_blocked else "Risk Level >= 75: High Threat Detected. Core Banking Freeze & Step-Up Verification Enforced."
        elif overall_risk_score >= 50.0:
            risk_tier = "AMBER"
            risk_label = "HIGH RISK - Suspicious Identity & Social Engineering"
            action_code = "STRONG_WARNING_VERIFY"
            adaptive_threshold_msg = "Risk Level 50-75: Strong Warning. Verification prompt enforced."
        elif overall_risk_score >= 30.0:
            risk_tier = "AMBER"
            risk_label = "SUSPICIOUS - Potential Voice Distortion / Unverified Call"
            action_code = "PASSIVE_WARNING"
            adaptive_threshold_msg = "Risk Level 30-50: Passive operator warning prompt."
        else:
            risk_tier = "GREEN"
            risk_label = "SAFE - Genuine Voice & Identity Verified"
            action_code = "ALLOW"
            adaptive_threshold_msg = "Risk Level < 30: Normal call flow active."

        if risk_tier == "RED_HIGH": risk_tier = "RED"

        # 4. Explainable AI Natural Language Reasoning ("Why Flagged?")
        explainability_bullets = []
        if network_blocked:
            explainability_bullets.append(f"Layer 0 Pre-Call Gate Triggered: {network_res.get('reason', 'Telecom signaling anomaly')}")
        elif network_risk_score >= 40.0:
            explainability_bullets.append(f"Network Telecom Signaling Anomaly ({network_risk_score}% risk)")
        if voice_authenticity_score >= 50.0:
            explainability_bullets.append(f"Synthetic speech indicators detected ({voice_authenticity_score}% synthetic probability)")
        if is_claimed and not is_match:
            explainability_bullets.append(f"Speaker similarity to claimed profile ({biometric_res.get('speaker_name')}) is low ({speaker_match_score}%)")
        elif not is_match:
            explainability_bullets.append("Caller identity unverified against registered Voice Identity Vault profiles")

        for tag in se_tags:
            explainability_bullets.append(f"Social engineering tactic detected: {tag}")

        if transaction_context_risk >= 50.0:
            explainability_bullets.append(f"High-value financial transaction context (₹{tx_amount:,.2f})")
        if replay_score >= 40.0:
            explainability_bullets.append("Playback channel acoustic distortion (Replay Attack suspected)")

        if not explainability_bullets:
            explainability_bullets.append("Normal human acoustic patterns and low contextual threat indicators.")

        explainability_summary = {
            "title": f"Why was this call assigned Risk Score {overall_risk_score}/100?",
            "reasons": explainability_bullets,
            "tactics_detected": se_tags,
            "adaptive_tier_guidance": adaptive_threshold_msg
        }

        return {
            "overall_risk_score": overall_risk_score,
            "risk_tier": risk_tier,
            "risk_label": risk_label,
            "action_code": action_code,
            "network_gate": network_res,
            "metric_breakdown": {
                "network_signaling_score": network_risk_score,
                "voice_authenticity_score": voice_authenticity_score,
                "speaker_match_score": speaker_match_score,
                "social_engineering_score": se_score,
                "transaction_context_risk": transaction_context_risk,
                "replay_risk_score": replay_score,
                "speaker_mismatch_penalty": round(speaker_mismatch_penalty, 1)
            },
            "explainability": explainability_summary,
            "biometric_details": biometric_res,
            "recommended_actions": [
                "Call Intercepted & Blocked before 2-way audio setup.",
                "Log SIP caller IP & CLI spoof details to National Cyber Crime Portal.",
                "Flag originating gateway trunk for telecom security review."
            ] if network_blocked else ([
                "Do NOT authorize financial transactions or disclose credentials on this call.",
                "Execute Step-Up Verification through an independent out-of-band communication channel.",
                "Escalate security audit record to Fraud Operations."
            ] if overall_risk_score >= 60 else [
                "Monitor call telemetry continuously.",
                "Verify identity if high-value action is requested."
            ])
        }
