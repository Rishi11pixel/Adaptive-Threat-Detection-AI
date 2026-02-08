from enum import Enum

class ThreatLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ProbabilisticThreatReasoner:
    def __init__(self):
        # Weighting factors (heuristic by design)
        self.vote_weight = 0.5
        self.persistence_weight = 0.3
        self.stability_weight = 0.2

    def compute_risk_score(self, votes, max_votes, persistence, stability):
        """
        votes: number of models flagging anomaly
        max_votes: total number of models
        persistence: float [0,1] (how long anomaly persists)
        stability: float [0,1] (alert smoothness)
        """

        vote_score = votes / max_votes

        risk_score = (
            self.vote_weight * vote_score +
            self.persistence_weight * persistence +
            self.stability_weight * stability
        )

        return round(min(risk_score, 1.0), 3)

    def map_to_threat_level(self, risk_score):
        if risk_score >= 0.75:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW
