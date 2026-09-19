import os
import json
from typing import Dict, Any, List

class AttackSimulator:
    """
    Attack Simulator Lab Engine.
    Provides pre-configured flagship SIH attack scenarios for live demonstration:
    - Scenario 1: Fake CFO Airport Fund Transfer Attack (Urgency + Secrecy + ₹30L Transfer)
    - Scenario 2: Fake Government Officer Confidential Data Attack
    - Scenario 3: Fake Bank Officer OTP Interception Attack
    - Scenario 4: Genuine CEO Routine Call
    """

    def __init__(self):
        self.scenarios = {
            "scenario_cfo_airport": {
                "id": "scenario_cfo_airport",
                "title": "🚨 Flagship Demo: Fake CFO - Airport Fund Transfer Attack",
                "attacker_type": "AI Voice Clone (ElevenLabs / RVC)",
                "claimed_speaker_id": "spk_cfo_ananya",
                "claimed_speaker_name": "Ananya Sen (CFO)",
                "target_victim": "Finance Operations Officer",
                "transaction_amount": 3000000.0, # ₹30 Lakhs
                "transcript": "Hi, I'm the CFO. I'm at the airport. I need you to transfer ₹30 lakh immediately to this new account. Don't call me back because I'm entering a meeting.",
                "sample_file": "data/sample_audio/sample_cloned_cfo_otp_attack.wav",
                "description": "Attacker impersonates CFO using a cloned synthetic voice, creating high time pressure (airport meeting), requesting ₹30L transfer to a new beneficiary, and explicitly instructing the employee not to call back."
            },
            "scenario_govt_officer": {
                "id": "scenario_govt_officer",
                "title": "🚨 Fake Government Official - Secrecy & Data Attack",
                "attacker_type": "Deepfake Voice Conversion",
                "claimed_speaker_id": "spk_cfo_ananya",
                "claimed_speaker_name": "Ministry Officer",
                "target_victim": "Enterprise Compliance Lead",
                "transaction_amount": 1000000.0,
                "transcript": "This is the Ministry Director. This is an urgent confidential matter. Send the security clearance document immediately and do not inform your supervisor.",
                "sample_file": "data/sample_audio/sample_cloned_ceo_impersonation.wav",
                "description": "Attacker uses authority pressure and secrecy instructions to extract confidential security documents."
            },
            "scenario_bank_otp": {
                "id": "scenario_bank_otp",
                "title": "🚨 Fake Bank Verification - Urgent OTP Social Engineering",
                "attacker_type": "Neural TTS Clone",
                "claimed_speaker_id": None,
                "claimed_speaker_name": "Unverified Bank Agent",
                "target_victim": "Bank Customer",
                "transaction_amount": 500000.0,
                "transcript": "Hello, your bank account will be blocked in 10 minutes due to policy bypass. State your OTP code right now to verify.",
                "sample_file": "data/sample_audio/sample_cloned_ceo_impersonation.wav",
                "description": "Attacker threatens immediate account blockage to trick victim into disclosing OTP credentials."
            },
            "scenario_genuine_ceo": {
                "id": "scenario_genuine_ceo",
                "title": "🟢 Genuine CEO Routine Verification Call",
                "attacker_type": "Genuine Human Speech",
                "claimed_speaker_id": "spk_ceo_rajesh",
                "claimed_speaker_name": "Rajesh Sharma (CEO)",
                "target_victim": "Executive Assistant",
                "transaction_amount": 50000.0,
                "transcript": "Hi, this is Rajesh. Please review the quarterly report when you get a chance.",
                "sample_file": "data/sample_audio/sample_genuine_ceo_english.wav",
                "description": "Authentic CEO call with natural human prosody, zero social engineering manipulation, and low transaction risk."
            }
        }

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return list(self.scenarios.values())

    def get_scenario(self, scenario_id: str) -> Dict[str, Any]:
        return self.scenarios.get(scenario_id, self.scenarios["scenario_cfo_airport"])
