from enum import Enum

class ThreatLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ThreatScorer:
    def __init__(self):
        # Tunable parameters (intentionally heuristic)
        self.low_threshold = 1
        self.medium_threshold = 2
        self.high_threshold = 3

    def score(self, votes, smoothed_alert):
        """
        votes: number of models flagging anomaly (0–3)
        smoothed_alert: boolean from temporal smoothing
        """

        # Conservative logic
        if votes >= self.high_threshold and smoothed_alert:
            return ThreatLevel.HIGH

        if votes >= self.medium_threshold:
            return ThreatLevel.MEDIUM

        return ThreatLevel.LOW
