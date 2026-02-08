import time

class AuditLogger:
    def log(self, threat_level, risk_score, action, reason):
        entry = {
            "timestamp": time.time(),
            "threat_level": threat_level.value,
            "risk_score": risk_score,
            "action": action.value,
            "reason": reason
        }

        print("[AUDIT]", entry)
        return entry
