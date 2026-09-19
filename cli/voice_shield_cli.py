#!/usr/bin/env python3
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sdk.voice_shield_sdk import VoiceShieldSDK

def main():
    parser = argparse.ArgumentParser(description="VoiceShield AI - Command Line Voice Integrity Scanner")
    parser.add_argument("file", help="Path to audio file (WAV/MP3) to analyze")
    parser.add_argument("--speaker-id", help="Claimed speaker identity ID (e.g. spk_ceo_rajesh)", default=None)
    parser.add_argument("--amount", help="Transaction amount (₹)", type=float, default=0.0)
    parser.add_argument("--transcript", help="Call transcript text context", default="")
    parser.add_argument("--server", help="VoiceShield server URL", default="http://localhost:8000")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Audio file '{args.file}' not found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" VOICE SHIELD AI - COMMAND LINE THREAT SCANNER")
    print(f"=======================================================")
    print(f"Scanning Target File: {args.file}")
    if args.speaker_id:
        print(f"Claimed Speaker ID:  {args.speaker_id}")
    if args.amount > 0:
        print(f"Transaction Context: ₹{args.amount:,.2f}")
    print(f"Server Endpoint:     {args.server}\n")

    try:
        sdk = VoiceShieldSDK(endpoint_url=args.server)
        res = sdk.verify_audio_file(
            file_path=args.file,
            claimed_speaker_id=args.speaker_id,
            transaction_amount=args.amount,
            transcript=args.transcript
        )

        risk = res["risk_summary"]
        score = risk["overall_risk_score"]
        tier = risk["risk_tier"]
        label = risk["risk_label"]

        tier_symbol = "🔴" if tier == "RED" else ("🟡" if tier == "AMBER" else "🟢")

        print(f" {tier_symbol} OVERALL IMPERSONATION RISK SCORE: {score} / 100 [{tier}]")
        print(f" Verdict:               {label}")
        print(f" Action Code:           {risk['action_code']}")
        print(f" Biometric Similarity:  {risk['metric_breakdown']['speaker_match_score']}%\n")

        print("--- 5-Vector Metric Breakdown ---")
        for k, v in risk["metric_breakdown"].items():
            print(f"  • {k}: {v}")

        print("\n--- Explainable AI Reasons ('Why Flagged?') ---")
        exp = risk.get("explainability", {})
        for reason in exp.get("reasons", []):
            print(f"  ✓ {reason}")

        print("\n--- Recommended Actions ---")
        for rec in risk["recommended_actions"]:
            print(f"  • {rec}")
        print("=======================================================\n")

    except Exception as e:
        print(f"Scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
