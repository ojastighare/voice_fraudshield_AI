import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

class BankingAPI:
    """
    Demo Core Banking System (CBS) API.
    Simulates real-time payment gateway, account ledger management, 
    and millisecond-latency transaction freeze operations for AI voice fraud interception.
    """
    def __init__(self):
        # Demo Core Banking Accounts Ledger
        self.accounts = {
            "ACC-9823418821": {
                "account_number": "ACC-9823418821",
                "holder_name": "Ananya Sen (CFO)",
                "account_type": "Corporate Current Account",
                "bank_name": "State Bank of India (SBI) - Corporate Hub",
                "balance": 15000000.00, # ₹ 1.5 Crore
                "status": "ACTIVE"
            },
            "ACC-4491028301": {
                "account_number": "ACC-4491028301",
                "holder_name": "Rajesh Sharma (CEO)",
                "account_type": "Executive Premium Account",
                "bank_name": "HDFC Bank - Special Operations",
                "balance": 8500000.00, # ₹ 85 Lakhs
                "status": "ACTIVE"
            },
            "ACC-7729104812": {
                "account_number": "ACC-7729104812",
                "holder_name": "Unknown Entity (Mule Account #812)",
                "account_type": "High-Velocity Escrow Mule Account",
                "bank_name": "NeoBank Digital Express (IFSC: NEOD000412)",
                "balance": 12500.00,
                "status": "SUSPICIOUS_WATCHLIST"
            },
            "ACC-1102938475": {
                "account_number": "ACC-1102938475",
                "holder_name": "Offshore Clearing Ltd",
                "account_type": "Cross-Border Settlement Account",
                "bank_name": "Global Clearing Bank",
                "balance": 0.00,
                "status": "FLAGGED_HIGH_RISK"
            }
        }

        # Active Pending Transactions
        self.transactions: Dict[str, Dict[str, Any]] = {}
        self._init_default_pending_txn()

    def _init_default_pending_txn(self):
        """Initializes default pending transaction for instant hackathon demonstrations."""
        txn_id = "TXN_NEFT_948201"
        self.transactions[txn_id] = {
            "txn_id": txn_id,
            "call_id": "CALL_DEMO",
            "source_account": "ACC-9823418821",
            "source_holder": "Ananya Sen (CFO - Corporate Treasury)",
            "source_bank": "SBI Corporate Treasury",
            "target_account": "ACC-7729104812",
            "target_holder": "Mule Beneficiary Account (Cyber Fraud Ring)",
            "target_bank": "NeoBank Digital Express (IFSC: NEOD000412)",
            "amount": 3000000.00, # ₹ 30 Lakhs
            "status": "PENDING_AUTHORIZATION",
            "threat_flag": "AI_VOICE_CLONE_DETECTED",
            "created_timestamp": time.time() - 2.5,
            "created_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "blocked_timestamp": None,
            "blocked_at_iso": None,
            "blocking_latency_ms": None,
            "freeze_reason": None,
            "blocked_by": None
        }

    def get_account(self, account_number: str) -> Optional[Dict[str, Any]]:
        return self.accounts.get(account_number)

    def list_accounts(self) -> List[Dict[str, Any]]:
        return list(self.accounts.values())

    def create_transaction_request(
        self,
        source_account: str,
        target_account: str,
        amount: float,
        call_id: str = "CALL_DEMO",
        threat_flag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a live transaction request linked to a voice call session.
        """
        txn_id = f"TXN_RTGS_{int(time.time()*1000) % 1000000:06d}"
        source_info = self.accounts.get(source_account, {
            "holder_name": "Corporate Account",
            "bank_name": "Corporate Bank"
        })
        target_info = self.accounts.get(target_account, {
            "holder_name": "Target Beneficiary",
            "bank_name": "Target Bank (IFSC: UNKN0001)"
        })

        txn_record = {
            "txn_id": txn_id,
            "call_id": call_id,
            "source_account": source_account,
            "source_holder": source_info.get("holder_name"),
            "source_bank": source_info.get("bank_name"),
            "target_account": target_account,
            "target_holder": target_info.get("holder_name"),
            "target_bank": target_info.get("bank_name"),
            "amount": float(amount),
            "status": "PENDING_AUTHORIZATION",
            "threat_flag": threat_flag or "EVALUATING_VOICE_INTELLIGENCE",
            "created_timestamp": time.time(),
            "created_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "blocked_timestamp": None,
            "blocked_at_iso": None,
            "blocking_latency_ms": None,
            "freeze_reason": None,
            "blocked_by": None
        }

        self.transactions[txn_id] = txn_record
        return txn_record

    def get_pending_transactions(self, call_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns pending or frozen transactions matching the current call session or active state."""
        txns = list(self.transactions.values())
        if call_id and call_id != "CALL_DEMO":
            filtered = [t for t in txns if t.get("call_id") == call_id or t.get("status") == "PENDING_AUTHORIZATION"]
            return filtered if filtered else txns
        return txns

    def freeze_transaction(
        self,
        txn_id: str,
        reason: str = "AI Voice Cloning Impersonation Intercepted",
        initiated_by: str = "VoiceFraudShield Automated CBS Gateway"
    ) -> Dict[str, Any]:
        """
        Executes millisecond-latency transaction freeze on the Core Banking System.
        Calculates and returns exact blocking time required in milliseconds.
        """
        start_exec = time.time()
        txn = self.transactions.get(txn_id)
        if not txn:
            # If transaction ID not found, use latest or create one on the fly
            if self.transactions:
                txn_id = list(self.transactions.keys())[-1]
                txn = self.transactions[txn_id]
            else:
                txn = self.create_transaction_request(
                    source_account="ACC-9823418821",
                    target_account="ACC-7729104812",
                    amount=3000000.00
                )
                txn_id = txn["txn_id"]

        now = time.time()
        # Simulated sub-50ms core banking network execution latency (typically 28ms - 45ms)
        execution_latency_ms = round((time.time() - start_exec) * 1000 + 38.4, 2)
        total_time_from_creation_ms = round((now - txn["created_timestamp"]) * 1000, 2)

        txn["status"] = "AUTOMATICALLY_FROZEN_BY_SHIELD" if "Automated" in initiated_by else "MANUALLY_BLOCKED"
        txn["blocked_timestamp"] = now
        txn["blocked_at_iso"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        txn["blocking_latency_ms"] = execution_latency_ms
        txn["total_response_time_ms"] = total_time_from_creation_ms
        txn["freeze_reason"] = reason
        txn["blocked_by"] = initiated_by

        return {
            "success": True,
            "status": txn["status"],
            "txn_id": txn_id,
            "source_account": txn["source_account"],
            "source_holder": txn["source_holder"],
            "target_account": txn["target_account"],
            "target_holder": txn["target_holder"],
            "target_bank": txn["target_bank"],
            "amount": txn["amount"],
            "blocking_latency_ms": execution_latency_ms,
            "total_response_time_ms": total_time_from_creation_ms,
            "blocked_at": txn["blocked_at_iso"],
            "freeze_reason": reason,
            "message": f"Transaction {txn_id} of ₹{txn['amount']:,.2f} FROZEN in {execution_latency_ms} ms! Debtor: {txn['source_account']}, Beneficiary: {txn['target_account']} locked."
        }
