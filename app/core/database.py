import os
import sqlite3
import json
import time
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "voice_fraud_shield.db")

class DatabaseManager:
    """
    SQLite Database Manager for Voice Fraud Shield.
    Stores Users, Voice Identity Vault Profiles, Call Sessions,
    Risk Events, Conversation NLP Events, and Audit Telemetry.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                organization_id TEXT DEFAULT 'ORG_DEFAULT',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Voice Identity Vault Profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                speaker_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                sample_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """)

            # 3. Call Sessions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                claimed_speaker_id TEXT,
                caller_number TEXT,
                receiver_number TEXT,
                overall_risk_score REAL DEFAULT 0.0,
                risk_tier TEXT DEFAULT 'GREEN',
                action_taken TEXT DEFAULT 'ALLOW',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Risk Events & Breakdown Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                synthetic_score REAL,
                speaker_similarity REAL,
                conversation_risk REAL,
                transaction_risk REAL,
                replay_risk REAL,
                overall_risk REAL,
                explainability_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            );
            """)

            # 5. Audit Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                user_or_system TEXT NOT NULL,
                anonymized_telemetry TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    def save_voice_profile(self, speaker_id: str, name: str, role: str, embedding: list) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            user_id = f"user_{speaker_id}"
            cursor.execute("INSERT OR REPLACE INTO users (id, name, role) VALUES (?, ?, ?)", (user_id, name, role))
            
            profile_id = f"prof_{speaker_id}"
            cursor.execute("""
            INSERT OR REPLACE INTO voice_profiles (id, user_id, speaker_id, name, role, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (profile_id, user_id, speaker_id, name, role, json.dumps(embedding)))
            conn.commit()
            return profile_id

    def log_call_risk_event(self, call_id: str, claimed_speaker_id: Optional[str], risk_fusion_data: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            score = risk_fusion_data.get("overall_risk_score", 0.0)
            tier = risk_fusion_data.get("risk_tier", "GREEN")
            action = risk_fusion_data.get("action_code", "ALLOW")

            cursor.execute("""
            INSERT OR REPLACE INTO calls (id, claimed_speaker_id, overall_risk_score, risk_tier, action_taken)
            VALUES (?, ?, ?, ?, ?)
            """, (call_id, claimed_speaker_id, score, tier, action))

            bd = risk_fusion_data.get("metric_breakdown", {})
            cursor.execute("""
            INSERT INTO risk_events (call_id, synthetic_score, speaker_similarity, conversation_risk, transaction_risk, replay_risk, overall_risk, explainability_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id,
                bd.get("voice_authenticity_score", 0.0),
                bd.get("speaker_match_score", 0.0),
                bd.get("social_engineering_score", 0.0),
                bd.get("transaction_context_risk", 0.0),
                bd.get("replay_risk_score", 0.0),
                score,
                json.dumps(risk_fusion_data.get("explainability", {}))
            ))
            conn.commit()
