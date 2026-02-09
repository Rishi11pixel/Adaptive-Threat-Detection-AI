import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time

from src.ai.probabilistic_reasoning import (ProbabilisticThreatReasoner,ThreatLevel)
from src.decision.decision_engine import DecisionEngine
from src.decision.audit_log import AuditLogger

reasoner = ProbabilisticThreatReasoner()
decision_engine = DecisionEngine()
logger = AuditLogger()

TOTAL_MODELS = 3  # pretend we have 3 ML models

def simulate_event(votes, persistence, stability, description):
    risk_score = reasoner.compute_risk_score(
        votes=votes,
        max_votes=TOTAL_MODELS,
        persistence=persistence,
        stability=stability
    )

    threat_level = reasoner.map_to_threat_level(risk_score)
    action, reason = decision_engine.decide(threat_level, risk_score)

    logger.log(threat_level, risk_score, action, reason)

    print(f"[SIMULATION] {description}")
    print(f"Votes={votes}, Persistence={persistence}, Stability={stability}")
    print(f"Risk={risk_score}, Threat={threat_level.value}, Action={action}\n")

# ------------------ SCENARIOS ------------------

def false_alarm_burst():
    print("\n--- False Alarm Burst ---")
    for _ in range(5):
        simulate_event(
            votes=random.choice([1, 2]),
            persistence=random.uniform(0.1, 0.3),
            stability=random.uniform(0.2, 0.4),
            description="Sensor glitch causing temporary spikes"
        )
        time.sleep(0.4)

def slow_burn_threat():
    print("\n--- Slow Burn Threat ---")
    persistence = 0.2
    for _ in range(6):
        persistence += random.uniform(0.1, 0.15)
        simulate_event(
            votes=2,
            persistence=min(persistence, 1.0),
            stability=random.uniform(0.5, 0.7),
            description="Gradual escalation over time"
        )
        time.sleep(0.6)

def high_confidence_short_threat():
    print("\n--- High Confidence Short Threat ---")
    simulate_event(
        votes=3,
        persistence=0.9,
        stability=0.85,
        description="Clear, short-duration threat"
    )

# ------------------ RUN ------------------

false_alarm_burst()
slow_burn_threat()
high_confidence_short_threat()
