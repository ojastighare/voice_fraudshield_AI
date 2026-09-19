import re
from typing import Dict, Any, List

class SocialEngineeringAnalyzer:
    """
    Layer 3: Conversation Risk & Social Engineering NLP Engine.
    Analyzes transcribed speech text to identify psychological manipulation tactics:
    - Urgency & time pressure ("immediately", "hurry", "entering a meeting", "jaldi")
    - Secrecy instructions ("don't tell anyone", "don't call me back", "gupt")
    - Authority pressure ("this is the CFO", "senior director here", "police", "customs")
    - High-value financial request ("transfer ₹30 lakh", "wire funds", "paise bhejo")
    - Policy bypass request ("override procedure", "skip dual authorization")
    - Credential & OTP requests ("share your OTP", "PIN", "password", "otp batao")
    - New beneficiary addition ("new account", "unregistered routing")
    - Digital arrest / Cyber coercion tactics
    """

    def __init__(self):
        # Tactic pattern definitions with Multilingual (English + Hindi / Hinglish + Indic Devanagari) support
        self.tactics = {
            "urgency": {
                "patterns": [
                    r"\b(immediately|urgent|urgently|hurry|right now|emergency|asap|fast|quick|airport|entering meeting|boarding|jaldi|turant|fatafat|abhi ke abhi|emergency hai|aapatkaal|der mat karo|foran)\b",
                    r"(जल्दी|तुरंत|फटाफट|अभी के अभी|इमरजेंसी|आपातकाल|शीघ्र|देर मत करो|फ़ौरन)"
                ],
                "weight": 24.0,
                "label": "Urgency & Time Pressure",
                "tag": "⚠ Urgency & Time Pressure"
            },
            "secrecy": {
                "patterns": [
                    r"\b(don't call (me )?back|do not call back|keep this (quiet|secret)|don't tell|nobody needs to know|private|confidential|kisi ko mat batana|phone mat karna|gupt|secret rakhna|kisi se share mat karo|call mat karna)\b",
                    r"(किसी को मत बताना|फोन मत करना|गुप्त|सीक्रेट|किसी से शेयर मत करो|कॉल मत करना|बात मत करना)"
                ],
                "weight": 26.0,
                "label": "Secrecy & Callback Bypass Instruction",
                "tag": "⚠ Secrecy / No-Callback"
            },
            "authority_pressure": {
                "patterns": [
                    r"\b(i am the (cfo|ceo|director|chairman|officer|manager)|senior executive|board member|command|main ceo bol raha hoon|cfo bol raha hoon|senior officer|adhikari|cbi|police|customs|tax officer|income tax|cyber cell)\b",
                    r"(सीईओ बोल रहा हूँ|सीएफओ बोल रहा हूँ|अधिकारी|सीनियर एग्जीक्यूटिव|डायरेक्टर|पुलिस|सीबीआई|कस्टम्स|इनकम टैक्स)"
                ],
                "weight": 20.0,
                "label": "Authority Impersonation & Coercion",
                "tag": "⚠ Authority Pressure"
            },
            "financial_request": {
                "patterns": [
                    r"\b(transfer|wire|send money|pay|deposit|rupees|lakh|crore|fund transfer|account transfer|paise bhejo|rakam|transfer karo|bhejo|bhugtan|advance payment|settle invoice|neft|rtgs|upi)\b",
                    r"(पैसे भेजो|ट्रांसफर करो|रकम|रुपये|लाख|करोड़|फंड ट्रांसफर|भुगतान|एनईएफटी|आरटीजीएस|यूपीआई)"
                ],
                "weight": 22.0,
                "label": "Unscheduled Financial Transaction Request",
                "tag": "⚠ Financial Transfer Request"
            },
            "policy_bypass": {
                "patterns": [
                    r"\b(bypass|override|skip|ignore (procedure|policy|verification)|special exception|authorize without|niyam chhod do|bina check kiye|process skip karo|dual authorization|compliance bypass)\b",
                    r"(नियम छोड़ दो|बिना चेक किए|प्रक्रिया छोड़ दो|बाईपास|वेरिफिकेशन छोड़ दो|अप्रूवल छोड़ो)"
                ],
                "weight": 24.0,
                "label": "Policy & Verification Bypass Attempt",
                "tag": "⚠ Policy Bypass Attempt"
            },
            "credential_otp_request": {
                "patterns": [
                    r"\b(otp|one time password|pin|password|security code|cvv|verification code|login details|otp batao|pin batao|password share karo|credential)\b",
                    r"(ओटीपी|ओटीपी बताओ|पिन बताओ|पासवर्ड बताओ|सिक्योरिटी कोड|वेरिफिकेशन कोड)"
                ],
                "weight": 32.0,
                "label": "Credential or OTP Interception Request",
                "tag": "⚠ OTP / Credential Harvest"
            },
            "new_beneficiary": {
                "patterns": [
                    r"\b(new account|different account|unregistered account|new beneficiary|vendor account|personal account|naya account|dusre account me|naya khata|third party account)\b",
                    r"(नया खाता|नया अकाउंट|दूसरे खाते में|नया लाभार्थी|पर्सनल अकाउंट)"
                ],
                "weight": 22.0,
                "label": "Unregistered / New Beneficiary Routing",
                "tag": "⚠ New Beneficiary Routing"
            }
        }

    def extract_transaction_context(self, transcript: str) -> Dict[str, Any]:
        """
        Extracts spoken monetary figures, currency mentions, and transaction intent
        directly from conversation transcript.
        """
        if not transcript or not transcript.strip():
            return {"amount": 0.0, "has_financial_intent": False, "context_risk": 5.0}

        text = transcript.lower()
        amount = 0.0
        has_intent = False

        # 1. Lakhs (e.g. "30 lakh", "30 lakhs", "25 lakh rupees", "30 लाख")
        lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|laakh|लाख)', text)
        if lakh_match:
            val = float(lakh_match.group(1))
            amount = val * 100000.0
            has_intent = True

        # 2. Crores (e.g. "2 crore", "1.5 crores", "करोड़")
        crore_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:crore|crores|करोड़)', text)
        if crore_match:
            val = float(crore_match.group(1))
            amount = max(amount, val * 10000000.0)
            has_intent = True

        # 3. Thousands (e.g. "50 thousand", "50 k", "50 हजार")
        thousand_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:thousand|k|हजार)', text)
        if thousand_match:
            val = float(thousand_match.group(1))
            amount = max(amount, val * 1000.0)
            has_intent = True

        # 4. Direct numbers with rupees / currency symbol (e.g. "₹50000", "50000 rupees", "rs 3000000")
        curr_match = re.search(r'(?:rs\.?|inr|₹|rupees|rupee)\s*(\d+[\d,]*)', text) or re.search(r'(\d+[\d,]*)\s*(?:rupees|rupee|inr)', text)
        if curr_match:
            num_str = curr_match.group(1).replace(',', '')
            try:
                num_val = float(num_str)
                if num_val >= 1000:
                    amount = max(amount, num_val)
                    has_intent = True
            except ValueError:
                pass

        # 5. Financial intent keywords without explicit digits
        financial_keywords = [
            "transfer", "wire", "fund transfer", "neft", "rtgs", "upi", "payment",
            "beneficiary", "settlement", "corporate payment", "budget approval",
            "invoice", "payee", "bhejo", "paise", "rupaye", "bhugtan", "khata"
        ]
        if any(kw in text for kw in financial_keywords):
            has_intent = True
            if amount == 0.0:
                amount = 50000.0 # Contextual standard amount

        # Compute contextual transaction risk score
        if amount >= 2000000: # >= 20 Lakhs
            context_risk = 95.0
        elif amount >= 500000: # >= 5 Lakhs
            context_risk = 75.0
        elif amount >= 100000: # >= 1 Lakh
            context_risk = 50.0
        elif amount >= 25000: # >= 25k
            context_risk = 30.0
        elif has_intent:
            context_risk = 20.0
        else:
            context_risk = 5.0 # Low baseline

        return {
            "amount": amount,
            "has_financial_intent": has_intent,
            "context_risk": round(context_risk, 1)
        }

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyzes conversation transcript and returns social engineering risk score (0-100),
        detected tactic tags, and confidence metrics.
        """
        if not transcript or not transcript.strip():
            return {
                "social_engineering_score": 0.0,
                "detected_tactics": [],
                "tactic_tags": [],
                "risk_level": "LOW",
                "matched_patterns_count": 0
            }

        text = transcript.lower().strip()
        detected_tactics = []
        tactic_tags = []
        raw_score = 0.0

        for key, t_info in self.tactics.items():
            matched_matches = []
            for pattern in t_info["patterns"]:
                matches = re.findall(pattern, text)
                if matches:
                    matched_matches.extend(matches)

            if matched_matches:
                raw_score += t_info["weight"]
                detected_tactics.append({
                    "tactic_id": key,
                    "label": t_info["label"],
                    "weight": t_info["weight"],
                    "matched_phrases": [str(m[0] if isinstance(m, tuple) else m) for m in matched_matches[:3]]
                })
                tactic_tags.append(t_info["tag"])

        # If conversation is present but no fraud tactics detected, return a clean minimal baseline (3.0%)
        if len(detected_tactics) == 0:
            se_score = 3.0
            risk_level = "LOW"
        else:
            se_score = float(min(round(raw_score, 1), 100.0))
            if se_score >= 60:
                risk_level = "CRITICAL"
            elif se_score >= 30:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

        return {
            "social_engineering_score": se_score,
            "detected_tactics": detected_tactics,
            "tactic_tags": tactic_tags,
            "risk_level": risk_level,
            "matched_patterns_count": len(detected_tactics)
        }
