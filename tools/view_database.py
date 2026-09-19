import sys
import os
import sqlite3
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import DB_PATH

def view_data():
    if not os.path.exists(DB_PATH):
        print(f"Database file '{DB_PATH}' not found yet. Run an analysis or attack simulation to populate database.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n=======================================================")
    print(" VOICE FRAUD SHIELD AI - BACKEND DATABASE VIEWER")
    print("=======================================================")
    print(f"Database File: {os.path.abspath(DB_PATH)}\n")

    # 1. Enrolled Voice Profiles (Voice Identity Vault)
    print("--- 1. VOICE IDENTITY VAULT (ENROLLED PROFILES) ---")
    cursor.execute("SELECT id, speaker_id, name, role, created_at FROM voice_profiles")
    profiles = cursor.fetchall()
    if profiles:
        for p in profiles:
            print(f" • [ID: {p['speaker_id']}] {p['name']} - {p['role']} (Enrolled: {p['created_at']})")
    else:
        print(" (No voice profiles enrolled yet)")

    # 2. Call Session History
    print("\n--- 2. RECENT CALL SESSION RISK LOGS ---")
    cursor.execute("SELECT id, claimed_speaker_id, overall_risk_score, risk_tier, action_taken, timestamp FROM calls ORDER BY timestamp DESC LIMIT 10")
    calls = cursor.fetchall()
    if calls:
        for c in calls:
            tier_symbol = "🔴" if c['risk_tier'] == 'RED' else ("🟡" if c['risk_tier'] == 'AMBER' else "🟢")
            claimed = c['claimed_speaker_id'] or 'Unverified'
            print(f" {tier_symbol} [Call ID: {c['id']}] Claimed: {claimed} | Risk: {c['overall_risk_score']}/100 [{c['risk_tier']}] | Action: {c['action_taken']} | Time: {c['timestamp']}")
    else:
        print(" (No call sessions logged yet)")

    # 3. Detailed Risk Events & 5-Vector Breakdown
    print("\n--- 3. RECENT RISK EVENTS & EXPLAINABLE AI BREAKDOWN ---")
    cursor.execute("SELECT call_id, synthetic_score, speaker_similarity, conversation_risk, transaction_risk, replay_risk, overall_risk, explainability_summary, created_at FROM risk_events ORDER BY id DESC LIMIT 3")
    events = cursor.fetchall()
    if events:
        for e in events:
            print(f"\n Event for Call [{e['call_id']}] (Score: {e['overall_risk']}/100):")
            print(f"   • Voice Authenticity: {e['synthetic_score']}% | Speaker Match: {e['speaker_similarity']}% | Social Eng: {e['conversation_risk']}% | Transaction: {e['transaction_risk']}% | Replay: {e['replay_risk']}%")
            if e['explainability_summary']:
                try:
                    exp = json.loads(e['explainability_summary'])
                    print("   • Explainable AI Reasons:")
                    for r in exp.get('reasons', []):
                        print(f"     ✓ {r}")
                except Exception:
                    pass
    else:
        print(" (No detailed risk events logged yet)")

    # 4. Enrolled JSON files on disk
    print("\n--- 4. DISK VOICEPRINT FILES ---")
    vp_dir = "data/enrolled_voiceprints"
    if os.path.exists(vp_dir):
        files = os.listdir(vp_dir)
        print(f" Folder '{vp_dir}': {files}")

    print("\n=======================================================\n")
    conn.close()

if __name__ == "__main__":
    view_data()
