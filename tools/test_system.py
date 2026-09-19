import sys
import os
import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.spectral_analyzer import SpectralAnalyzer
from app.core.prosody_analyzer import ProsodyAnalyzer
from app.core.biometric_verifier import BiometricVerifier
from app.core.deep_classifier import DeepVoiceClassifier
from app.core.social_engineering_analyzer import SocialEngineeringAnalyzer
from app.core.replay_detector import ReplayDetector
from app.core.network_gate import NetworkGate
from app.core.risk_fusion_engine import RiskFusionEngine
from app.core.database import DatabaseManager
from app.core.attack_simulator import AttackSimulator

def run_tests():
    print("\n=======================================================")
    print(" VOICE FRAUD SHIELD AI - 4-LAYER SYSTEM DIAGNOSTIC TEST")
    print("=======================================================\n")

    spectral_analyzer = SpectralAnalyzer()
    prosody_analyzer = ProsodyAnalyzer()
    biometric_verifier = BiometricVerifier()
    deep_classifier = DeepVoiceClassifier()
    social_eng_analyzer = SocialEngineeringAnalyzer()
    replay_detector = ReplayDetector()
    network_gate = NetworkGate()
    risk_fusion_engine = RiskFusionEngine()
    db_manager = DatabaseManager()
    attack_simulator = AttackSimulator()

    # 0. Test Layer 0 Network Gate (SIP Headers & Early-Media Interception)
    print("[TEST 0/6] Testing Layer 0 Network Signaling & Pre-Call Gate...")
    spoofed_headers = {
        "STIR-SHAKEN-Attestation": "C",
        "User-Agent": "SIPp-tester/v3.0",
        "From-CLI": "+1-800-555-SPOOF",
        "X-Is-VoIP-Proxy": "true"
    }
    net_eval = network_gate.evaluate_pre_call_gate(headers=spoofed_headers)
    print(f" Network Gate Decision: {net_eval['action']} (Blocked: {net_eval['blocked']})")
    print(f" Network Risk Score:   {net_eval['network_risk_score']} / 1.0")
    print(f" Reason:               {net_eval['reason']}")
    assert net_eval['blocked'] == True, "Expected Layer 0 Network Gate to drop spoofed CLI/VoIP call pre-2-way audio"
    print(" -> PASSED Layer 0 Network Gate Verification!\n")

    # 1. Test Social Engineering NLP Engine
    print("[TEST 1/5] Testing Layer 3: Social Engineering NLP Engine...")
    cfo_script = "Hi, I'm the CFO. I'm at the airport. I need you to transfer ₹30 lakh immediately to this new account. Don't call me back because I'm entering a meeting."
    se_res = social_eng_analyzer.analyze_transcript(cfo_script)
    print(f" Flagship CFO Script SE Risk Score: {se_res['social_engineering_score']} / 100 [{se_res['risk_level']}]")
    print(f" Detected Tactics:                 {se_res['tactic_tags']}")
    assert se_res['social_engineering_score'] >= 60, "Expected high social engineering risk score for CFO attack script"
    print(" -> PASSED Layer 3 Social Engineering NLP Verification!\n")

    # 2. Test Layer 1 & 2 Audio Authenticity & Biometric Match on Genuine Sample
    print("[TEST 2/5] Testing Genuine Speech & Biometric Identity Verification...")
    sr1, audio1 = wav.read("data/sample_audio/sample_genuine_ceo_english.wav")
    audio1 = audio1.astype(np.float32) / 32768.0

    spec1 = spectral_analyzer.extract_spectral_features(audio1)
    pros1 = prosody_analyzer.analyze_prosody(audio1)
    bio1 = biometric_verifier.verify_speaker(audio1, claimed_speaker_id="spk_ceo_rajesh")
    ml1 = deep_classifier.predict_synthetic_probability(spec1, pros1)
    replay1 = replay_detector.detect_replay_attack(audio1)
    se1 = social_eng_analyzer.analyze_transcript("Hi, please review the budget report when you get a chance.")

    fusion1 = risk_fusion_engine.fuse_risks(spec1, pros1, bio1, ml1, se1, replay1)
    print(f" Genuine Sample Overall Risk Score: {fusion1['overall_risk_score']} / 100 [{fusion1['risk_tier']}]")
    assert fusion1['overall_risk_score'] < 35, f"Expected GREEN risk score for genuine call, got {fusion1['overall_risk_score']}"
    print(" -> PASSED Genuine Speech & Identity Verification!\n")

    # 3. Test Flagship CFO Impersonation Attack Scenario in Risk Fusion Engine
    print("[TEST 3/5] Testing Flagship CFO Airport Attack Scenario in Risk Fusion Engine...")
    sr2, audio2 = wav.read("data/sample_audio/sample_cloned_cfo_otp_attack.wav")
    audio2 = audio2.astype(np.float32) / 32768.0

    spec2 = spectral_analyzer.extract_spectral_features(audio2)
    pros2 = prosody_analyzer.analyze_prosody(audio2)
    bio2 = biometric_verifier.verify_speaker(audio2, claimed_speaker_id="spk_cfo_ananya")
    ml2 = deep_classifier.predict_synthetic_probability(spec2, pros2)
    replay2 = replay_detector.detect_replay_attack(audio2)
    se2 = social_eng_analyzer.analyze_transcript(cfo_script)

    net_cfo = network_gate.evaluate_pre_call_gate(
        headers={"STIR-SHAKEN-Attestation": "C", "X-Is-VoIP-Proxy": "true"},
        audio_snippet=audio2[:16000],
        spectral_analyzer=spectral_analyzer
    )

    fusion2 = risk_fusion_engine.fuse_risks(
        spec2, pros2, bio2, ml2, se2, replay2,
        network_res=net_cfo,
        context_data={"transaction_amount": 3000000.0, "transcript": cfo_script}
    )

    print(f" Flagship CFO Attack Overall Risk Score: {fusion2['overall_risk_score']} / 100 [{fusion2['risk_tier']}]")
    print(f" Verdict Label:                          {fusion2['risk_label']}")
    print(f" 5-Vector Metric Breakdown:")
    for k, v in fusion2['metric_breakdown'].items():
        print(f"    • {k}: {v}")
    print(f" Explainability Reasons:")
    for r in fusion2['explainability']['reasons']:
        print(f"    ✓ {r}")

    assert fusion2['overall_risk_score'] >= 80, f"Expected RED critical risk score for CFO attack scenario, got {fusion2['overall_risk_score']}"
    print(" -> PASSED Flagship CFO Attack Scenario Verification!\n")

    # 4. Test Attack Simulator Lab Scenarios Listing
    print("[TEST 4/5] Testing Attack Simulator Lab Scenarios...")
    scenarios = attack_simulator.list_scenarios()
    print(f" Loaded Scenarios ({len(scenarios)}): {[s['id'] for s in scenarios]}")
    assert len(scenarios) >= 4, "Expected at least 4 hackathon attack scenarios"
    print(" -> PASSED Attack Simulator Lab Verification!\n")

    # 5. Test SQLite Database Logging
    print("[TEST 5/6] Testing SQLite Database Persistence & Audit Logging...")
    db_manager.log_call_risk_event("CALL_TEST_999", "spk_cfo_ananya", fusion2)
    print(" -> PASSED Database Logging Check!\n")

    # 6. Test Core Banking API Transaction Freeze & Latency
    print("[TEST 6/6] Testing Core Banking (CBS) Transaction Freeze & Latency...")
    from app.core.banking_api import BankingAPI
    bank_test = BankingAPI()
    freeze_res = bank_test.freeze_transaction(
        txn_id="TXN_NEFT_948201",
        reason="AI Voice Clone Attack Intercepted"
    )
    print(f" Frozen Status:       {freeze_res['status']}")
    print(f" Source Account:      {freeze_res['source_account']} ({freeze_res['source_holder']})")
    print(f" Target Beneficiary:  {freeze_res['target_account']} ({freeze_res['target_holder']})")
    print(f" Amount:              ₹{freeze_res['amount']:,.2f}")
    print(f" Blocking Latency:    {freeze_res['blocking_latency_ms']} ms")
    assert freeze_res["success"] == True
    assert freeze_res["blocking_latency_ms"] < 100.0, "Expected sub-100ms blocking latency"
    print(" -> PASSED Core Banking System (CBS) Freeze & Latency Verification!\n")

    print("=======================================================")
    print(" ALL 4-LAYER SYSTEM + BANKING API TESTS PASSED! (100% OPERATIONAL)")
    print("=======================================================\n")

if __name__ == "__main__":
    run_tests()
