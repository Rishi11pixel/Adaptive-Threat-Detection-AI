import time
from enum import Enum
from src.ai.probabilistic_reasoning import ThreatLevel

class Action(Enum):
    IGNORE = "IGNORE"
    MONITOR = "MONITOR"
    ALERT_ANALYST = "ALERT_ANALYST"
    ESCALATE = "ESCALATE"

class DecisionEngine:
    def __init__(self):
        self.last_escalation_time = None
        self.cooldown_seconds = 60  # prevent alert storms

    def decide(self, threat_level, risk_score):
        current_time = time.time()

        # Cooldown check
        if self.last_escalation_time:
            if current_time - self.last_escalation_time < self.cooldown_seconds:
                return Action.MONITOR, "Cooldown active"

        if threat_level == ThreatLevel.HIGH and risk_score > 0.85:
            self.last_escalation_time = current_time
            return Action.ESCALATE, "High confidence threat detected"

        if threat_level == ThreatLevel.MEDIUM:
            return Action.ALERT_ANALYST, "Moderate threat – human review required"

        if threat_level == ThreatLevel.LOW:
            return Action.MONITOR, "Low risk – monitoring only"

        return Action.IGNORE, "No action required"
